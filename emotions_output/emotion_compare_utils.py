"""Shared helpers for comparing annotated emotion labels with VLM answers.

Used by
* compare_emotions_data_out.ipynb  (single output file)
* compare_emotions_models.ipynb    (four model files)

Every VLM table is first converted to a wide table ``filename, model, Q1..Q8``;
all further functions work on that table (or the long table derived from it), so they
handle any number of models.
"""

import ast
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
QIDS = [f"Q{i}" for i in range(1, 9)]
TARGETS = ["polarity", "polarity", "negative", "negative", "negative", "positive", "positive", "positive"]
STYLES = ["open", "constrained", "open", "constrained", "constrained+cues", "open", "constrained", "constrained+cues"]


def load_questions(path):
    """questions.txt contains a python list assignment -> parse it safely."""
    txt = Path(path).read_text()
    questions = ast.literal_eval(txt[txt.index("["):])
    assert len(questions) == len(QIDS), f"expected {len(QIDS)} questions, got {len(questions)}"
    return pd.DataFrame({"qid": QIDS, "target": TARGETS, "style": STYLES, "question": questions}).set_index("qid")


QMETA = pd.DataFrame({"qid": QIDS, "target": TARGETS, "style": STYLES}).set_index("qid")


def qlabel(q):
    return f"{q}\n{QMETA.loc[q, 'target'][:3]}/{QMETA.loc[q, 'style']}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_labels(path):
    lab = pd.read_csv(path)
    lab["filename"] = lab["filename"].str.strip()
    lab["subfolder"] = lab["subfolder"].fillna("(none)")
    assert not lab["filename"].duplicated().any(), "duplicate filenames in labels"
    return lab


def parse_vqa_list(s, n=len(QIDS)):
    """Return a list of length n; entries are None when the answer is missing/empty."""
    try:
        lst = ast.literal_eval(s) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        lst = []
    lst = [a if isinstance(a, str) and a.strip() else None for a in lst]
    return (lst + [None] * n)[:n]


def load_vqa(path, model_name=None):
    """Wide table: filename, model, Q1..Q8 (raw answers, None = missing)."""
    raw = pd.read_csv(path)
    out = pd.DataFrame({"filename": raw["filename"].map(os.path.basename)})
    if "model" in raw.columns and model_name is None:
        out["model"] = raw["model"].values
    else:
        out["model"] = model_name or Path(path).stem
    answers = raw["vqa"].map(parse_vqa_list)
    for i, q in enumerate(QIDS):
        out[q] = answers.map(lambda l, i=i: l[i])
    assert not out.duplicated(["model", "filename"]).any(), f"duplicate filenames in {path}"
    return out


def join_labels(vqa, labels):
    only_lab = set(labels.filename) - set(vqa.filename)
    only_vqa = set(vqa.filename) - set(labels.filename)
    print(f"labels: {len(labels)}  vqa rows: {len(vqa)}  "
          f"filenames only in labels: {len(only_lab)}  only in vqa: {len(only_vqa)}")
    return vqa.merge(labels, on="filename", how="inner", validate="many_to_one")


# ---------------------------------------------------------------------------
# Normalisation
#
# Every normaliser returns (category, rule) so that the mapping can be audited.
# Rules, in order of precedence:
#   missing   - no answer
#   exact     - the whole answer is a category ("Yes.", "Negative", "DK")
#   first     - the first word is a category ("No, the picture ...")
#   answer_is - "... the answer is: yes"
#   kw_*      - keyword rules for free-text answers (see below)
#   unparsed  - nothing matched
# ---------------------------------------------------------------------------
POLARITY_CATS = ["positive", "negative", "both", "none", "dk"]
YESNO_CATS = ["yes", "no", "dk"]

# hedging / can't tell
DK_PAT = re.compile(
    r"^(dk|i don'?t know|don'?t know|do not know|unsure|not sure)$"
    r"|difficult to (definitively |accurately |clearly )?(determine|assess|say|tell|judge|interpret)"
    r"|(can'?t|cannot|unable to) (be )?(definitively |accurately )?(determine|assess|tell|judge)"
    r"|cannot be (definitively |accurately )?(determined|assessed)"
    r"|(not|n'?t) (provide )?(enough|sufficient) (context|information)"
    r"|too (blurry|small|dark)"
    r"|unclear (if|whether)"
    r"|(is|remains) subjective|depends on (the )?(individual|context|viewer)")
# "without (more) context" is hedging, not a negation of the emotion
CONTEXT_HEDGE = re.compile(r"without (any |more |additional |further |the )?(context|information|visual cues)[^.;]*")

# words that signal emotional polarity in free text
NEG_WORDS = (r"negativ\w*|somber|sombre|sad\w*|tense|tension|conflict\w*|fear\w*|anger|angry|distress\w*"
             r"|grief|danger\w*|chao\w*|destruct\w*|devastat\w*|violen\w*|frustrat\w*|concern\w*|pain\w*"
             r"|loss|tragic|alarm\w*|disappoint\w*|melanchol\w*|solemn\w*|unrest|hardship|anxi\w*|worr\w*"
             r"|disapprov\w*|critical|shock\w*|upset")
POS_WORDS = (r"positiv\w*|happ\w*|joy\w*|celebrat\w*|cheerful|smil\w*|laugh\w*|warm|friendly|triumph\w*"
             r"|delight\w*|optimis\w*|enjoy\w*|pleasant|uplifting|cheer\w*|excite\w*|excitement|contentment"
             r"|gratitude|pride|proud|love|affection\w*")
NEG_RE = re.compile(rf"\b({NEG_WORDS})\b")
POS_RE = re.compile(rf"\b({POS_WORDS})\b")
# negator followed (in the same clause, <= 60 chars) by an emotion word
NEGATOR = r"\b(not|n't|no|never|without|lack\w*|rather than|nor|neither|absence of|free of|instead of)\b"

# "no emotion at all" (deliberately NOT "does not show any positive emotion" - that says nothing about negative)
_ADJ = r"( explicit| clear| obvious| particular| strong| specific| discernible| overt| apparent| visible)?"
NONE_RE = re.compile(
    r"(does not|doesn'?t|do not|did not|is not)( explicitly| clearly| really| necessarily| inherently)?"
    rf" (contain|show|convey|display|depict|have|evoke|present|express)( any)?{_ADJ}( positive or negative| positive and negative)? emotion"
    rf"|no{_ADJ}( positive or negative)? emotion|there is no{_ADJ}( positive or negative)? emotion"
    r"|without any (clear |explicit |obvious )?emotional content"
    r"|neither positive nor negative|neither overtly positive nor negative|emotionally neutral")
NEUTRAL_RE = re.compile(r"\bneutral\b")
HOWEVER_RE = re.compile(r"\b(however|but)\b(.*)")
BOTH_RE = re.compile(r"both positive and negative|mix(ed|ture)? of (emotions|positive and negative)|mixed emotion")
# explicit conclusion: "Therefore, the (correct) answer is: negative", "Answer: yes", 'the answer would be "none"'
_ANSWER_RE = re.compile(r"\banswer(?: is| would be)?(?: that)?\s*:?\s*[\"'*]*\s*(?:the picture shows\s*[\"'*]*)?"
                        r"(yes|no|dk|positive|negative|both|none)\b")


def _explicit_answer(c):
    """Category of the LAST explicit 'answer is ...' in the text, or None."""
    hits = _ANSWER_RE.findall(c)
    return hits[-1] if hits else None


def _isna(s):
    return s is None or (isinstance(s, float) and math.isnan(s))


def _clean(s):
    s = s.lower().replace("*", "").replace("’", "'").strip()
    return re.sub(r"^[\s\"'`]+|[\s\"'`.!:,]+$", "", s)


def _first_word(c):
    return re.split(r"[\s.,;:!]+", c, maxsplit=1)[0]


def _negated(c, word_re_src):
    return re.search(rf"{NEGATOR}[^.;]{{0,60}}?\b({word_re_src})\b", c) is not None


def norm_polarity(s):
    """Q1/Q2 -> positive | negative | both | none | dk | unparsed | missing."""
    if _isna(s):
        return "missing", "missing"
    c = _clean(s)
    alias = {"neutral": "none", "mixed": "both", "no": "none"}  # "No" to "any positive or negative?" = none
    if c in POLARITY_CATS or c in alias:
        return alias.get(c, c), "exact"
    w = _first_word(c)
    if w in POLARITY_CATS or w in ("neutral", "mixed"):
        return alias.get(w, w), "first"
    a = _explicit_answer(c)
    if a and a != "yes":
        return {"no": "none"}.get(a, a), "answer_is"
    if BOTH_RE.search(c):
        return "both", "kw_both"
    m = NONE_RE.search(c)
    if m:
        # "... shows no emotion. However, the text suggests a negative topic" -> follow the 'however' clause
        h = HOWEVER_RE.search(c[m.end():])
        pol = _first_polarity_keyword(re.split(r"[.;!]", h.group(2), maxsplit=1)[0]) if h else None
        return (pol, "kw_none_however") if pol else ("none", "kw_none")
    if DK_PAT.search(c):
        return "dk", "kw_dk"
    first_sentence = re.split(r"(?<=[.;!])\s", c, maxsplit=1)[0]
    if NEUTRAL_RE.search(first_sentence) and not _first_polarity_keyword(first_sentence):
        return "none", "kw_neutral"
    pol = _first_polarity_keyword(c)
    if pol:
        return pol, "kw_polarity"
    if NEUTRAL_RE.search(c):
        return "none", "kw_neutral"
    return "unparsed", "unparsed"


# a negator followed by an emotion word, including chains such as "no clear signs of positive or negative emotions"
_KW = rf"({NEG_WORDS}|{POS_WORDS})"
NEGATED_CHAIN = re.compile(rf"{NEGATOR}[^.;]{{0,60}}?\b{_KW}\b(\s*(,|or|and|nor|/)\s*(\w+\s+)?{_KW}\b)*")


def _first_polarity_keyword(c):
    """First non-negated emotional keyword wins ("no clear indication of positive ...; rather tension" -> negative)."""
    c2 = NEGATED_CHAIN.sub(" ", c)
    m_neg, m_pos = NEG_RE.search(c2), POS_RE.search(c2)
    if m_neg and (not m_pos or m_neg.start() < m_pos.start()):
        return "negative"
    if m_pos:
        return "positive"
    return None


def norm_yesno(s, target):
    """Q3-Q8 -> yes | no | dk | unparsed | missing.

    target: 'negative' or 'positive' (what the question asks about). Free-text rules:
      1. kw_negated : the asked-for emotion is negated   ("does not show any negative ...") -> no
      2. kw_dk      : hedging                            ("difficult to determine ...")      -> dk
      3. kw_affirm  : the asked-for emotion is mentioned ("... indicating a positive tone")  -> yes
      4. kw_opposite: only the opposite emotion / neutrality is mentioned                    -> no
    """
    if _isna(s):
        return "missing", "missing"
    c = _clean(s)
    if c in YESNO_CATS:
        return c, "exact"
    w = _first_word(c)
    if w in YESNO_CATS:
        return w, "first"
    a = _explicit_answer(c)
    if a in YESNO_CATS:
        return a, "answer_is"
    if c in ("i don't know", "don't know"):
        return "dk", "exact"
    own, other = (NEG_WORDS, POS_WORDS) if target == "negative" else (POS_WORDS, NEG_WORDS)
    c_noctx = CONTEXT_HEDGE.sub(" ", c)
    if _negated(c_noctx, own):
        return "no", "kw_negated"
    if DK_PAT.search(c):
        return "dk", "kw_dk"
    if re.search(rf"\b({own})\b", c_noctx):
        return "yes", "kw_affirm"
    if re.search(rf"\b({other})\b", c_noctx) or NONE_RE.search(c_noctx):
        return "no", "kw_opposite"
    return "unparsed", "unparsed"


def normalise(df, qmeta=QMETA):
    """Add Qx_norm and Qx_rule columns."""
    out = df.copy()
    for q in QIDS:
        target = qmeta.loc[q, "target"]
        f = norm_polarity if target == "polarity" else (lambda s, t=target: norm_yesno(s, t))
        res = out[q].map(f)
        out[q + "_norm"] = res.map(lambda r: r[0])
        out[q + "_rule"] = res.map(lambda r: r[1])
    return out


def rule_counts(df):
    """How many answers were normalised by which rule (per model and question)."""
    t = pd.concat({q: df.groupby("model")[q + "_rule"].value_counts() for q in QIDS}, names=["qid"])
    return t.unstack(fill_value=0)


def mapping_review(df, rules=None):
    """Distinct raw answers that are not exact categories, with their normalised value and rule."""
    rows = []
    for q in QIDS:
        sub = df[["model", q, q + "_norm", q + "_rule"]].dropna(subset=[q])
        sub = sub[~sub[q + "_rule"].isin(["exact", "missing"])]
        if rules is not None:
            sub = sub[sub[q + "_rule"].isin(rules)]
        sub.columns = ["model", "raw", "normalised", "rule"]
        sub.insert(1, "qid", q)
        rows.append(sub)
    r = pd.concat(rows)
    return (r.value_counts().rename("n").reset_index()
            .sort_values(["model", "qid", "rule", "normalised"]).reset_index(drop=True))


def answer_distribution(df):
    order = POLARITY_CATS + ["yes", "no", "unparsed", "missing"]
    t = pd.concat({q: df.groupby("model")[q + "_norm"].value_counts() for q in QIDS}, names=["qid"])
    t = t.unstack(fill_value=0)
    return t[[c for c in order if c in t.columns]]


# ---------------------------------------------------------------------------
# Ground truth and long table
# ---------------------------------------------------------------------------
TRUTH = {
    "polarity": {"positive": "positive", "negative": "negative", "neutral": "none"},
    "negative": {"positive": "no", "negative": "yes", "neutral": "no"},
    "positive": {"positive": "yes", "negative": "no", "neutral": "no"},
}
LABEL_ORDER = ["negative", "neutral", "positive", "unclear"]


def _long_part(df, qid, target, style, pred_col, raw_col=None):
    part = df[["model", "filename", "emotion", "subfolder"]].copy()
    part["qid"], part["target"], part["style"] = qid, target, style
    part["raw"] = df[raw_col] if raw_col else None
    part["pred"] = df[pred_col]
    part["truth"] = df["emotion"].map(TRUTH[target])  # NaN for 'unclear'
    part["scored"] = part["truth"].notna() & (part["pred"] != "missing")
    part["correct"] = np.where(part["scored"], part["pred"] == part["truth"], np.nan)
    return part


def to_long(df, qmeta=QMETA):
    """One row per model x image x question."""
    return pd.concat([_long_part(df, q, qmeta.loc[q, "target"], qmeta.loc[q, "style"], q + "_norm", q)
                      for q in QIDS], ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def prf(truth, pred, label):
    tp = np.sum((pred == label) & (truth == label))
    npred, ntrue = np.sum(pred == label), np.sum(truth == label)
    p = tp / npred if npred else 0.0
    r = tp / ntrue if ntrue else 0.0
    return p, r, _f1(p, r)


def accuracy(truth, pred):
    return float(np.mean(truth == pred))


def macro_f1(truth, pred):
    return float(np.mean([prf(truth, pred, l)[2] for l in np.unique(truth)]))


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in np.union1d(a, b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def bootstrap_ci(truth, pred, fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(truth)
    stats = [fn(truth[idx], pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.nanpercentile(stats, [2.5, 97.5])


def question_metrics(long, by=("model", "qid"), ci=True, n_boot=1000, seed=42):
    rows = []
    for key, g in long.groupby(list(by), sort=False):
        s = g[g["scored"]]
        t, p = s["truth"].to_numpy(), s["pred"].to_numpy()
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update(target=g["target"].iloc[0], style=g["style"].iloc[0],
                 n_images=len(g), n_scored=len(s), n_missing=int((g["pred"] == "missing").sum()),
                 dk_rate=np.mean(p == "dk") if len(p) else np.nan,
                 unparsed_rate=np.mean(p == "unparsed") if len(p) else np.nan)
        if len(s):
            r["accuracy"] = accuracy(t, p)
            committed = ~np.isin(p, ["dk", "unparsed"])  # accuracy when the model commits to an answer
            r["acc_answered"] = accuracy(t[committed], p[committed]) if committed.any() else np.nan
            r["macro_f1"] = macro_f1(t, p)
            r["kappa"] = cohen_kappa(t, p)
            if ci:
                r["acc_lo"], r["acc_hi"] = bootstrap_ci(t, p, accuracy, n_boot, seed)
                r["f1_lo"], r["f1_hi"] = bootstrap_ci(t, p, macro_f1, n_boot, seed)
            if r["target"] != "polarity":
                r["yes_precision"], r["yes_recall"], r["yes_f1"] = prf(t, p, "yes")
        rows.append(r)
    return pd.DataFrame(rows)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj
    return out


PROMPT_PAIRS = [("Q1", "Q2"), ("Q3", "Q4"), ("Q3", "Q5"), ("Q4", "Q5"), ("Q6", "Q7"), ("Q6", "Q8"), ("Q7", "Q8")]


def mcnemar_prompts(long, pairs=PROMPT_PAIRS):
    """Paired comparison of question variants within each model (Holm-corrected within model)."""
    rows = []
    sc = long[long.scored]
    for m in sc.model.unique():
        w = sc[sc.model == m].pivot(index="filename", columns="qid", values="correct")
        for q1, q2 in pairs:
            d = w[[q1, q2]].dropna().astype(bool)
            b, c = int((d[q1] & ~d[q2]).sum()), int((~d[q1] & d[q2]).sum())
            rows.append({"model": m, "pair": f"{q1} vs {q2}", "n": len(d),
                         "acc_first": d[q1].mean(), "acc_second": d[q2].mean(),
                         "only_first_correct": b, "only_second_correct": c, "p_value": mcnemar_exact(b, c)})
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("model")["p_value"].transform(holm)
    return t


def mcnemar_models(long):
    """Paired comparison of models per question on images answered by both (Holm-corrected within question)."""
    rows = []
    sc = long[long.scored]
    models = list(sc.model.unique())
    for q in sc.qid.unique():
        w = sc[sc.qid == q].pivot(index="filename", columns="model", values="correct")
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                d = w[[m1, m2]].dropna().astype(bool)
                b, c = int((d[m1] & ~d[m2]).sum()), int((~d[m1] & d[m2]).sum())
                rows.append({"qid": q, "model_a": m1, "model_b": m2, "n": len(d),
                             "acc_a": d[m1].mean(), "acc_b": d[m2].mean(),
                             "only_a_correct": b, "only_b_correct": c, "p_value": mcnemar_exact(b, c)})
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("qid")["p_value"].transform(holm)
    return t


# ---------------------------------------------------------------------------
# Consistency between questions
# ---------------------------------------------------------------------------
DERIVED = {"D(Q3,Q6)": ("Q3", "Q6"), "D(Q4,Q7)": ("Q4", "Q7"), "D(Q5,Q8)": ("Q5", "Q8")}
POL_COLS = ["Q1", "Q2"] + list(DERIVED)


def derive_polarity(neg, pos):
    if "missing" in (neg, pos):
        return "missing"
    return {("yes", "no"): "negative", ("no", "yes"): "positive",
            ("no", "no"): "none", ("yes", "yes"): "both"}.get((neg, pos), "dk")


def add_derived(df):
    df = df.copy()
    for name, (qn, qp) in DERIVED.items():
        df[name + "_norm"] = [derive_polarity(a, b) for a, b in zip(df[qn + "_norm"], df[qp + "_norm"])]
    return df


def derived_long(df):
    return pd.concat([_long_part(df, name, "polarity", "derived", name + "_norm") for name in DERIVED],
                     ignore_index=True)


def pairwise_agreement(df, cols, model):
    d = df[df.model == model]
    agree = pd.DataFrame(index=cols, columns=cols, dtype=float)
    kappa = agree.copy()
    for a in cols:
        for b in cols:
            x, y = d[a + "_norm"], d[b + "_norm"]
            ok = (x != "missing") & (y != "missing")
            agree.loc[a, b] = np.mean(x[ok] == y[ok]) if ok.any() else np.nan
            kappa.loc[a, b] = cohen_kappa(x[ok], y[ok])
    return agree, kappa


def contradictions(df):
    rows = []
    for m, d in df.groupby("model", sort=False):
        for pol in ["Q1", "Q2"]:
            for neg, pos in DERIVED.values():
                P, N, Y = d[pol + "_norm"], d[neg + "_norm"], d[pos + "_norm"]
                rows.append({"model": m, "polarity_q": pol, "neg_q": neg, "pos_q": pos,
                             "pol=positive & neg=yes": int(((P == "positive") & (N == "yes")).sum()),
                             "pol=negative & pos=yes": int(((P == "negative") & (Y == "yes")).sum()),
                             "pol=none & (neg|pos)=yes": int(((P == "none") & ((N == "yes") | (Y == "yes"))).sum()),
                             "pol=positive & pos=no": int(((P == "positive") & (Y == "no")).sum()),
                             "pol=negative & neg=no": int(((P == "negative") & (N == "no")).sum())})
    return pd.DataFrame(rows)


def subfolder_accuracy(long, model):
    s = long[(long.model == model) & long.scored]
    acc = s.pivot_table(index="subfolder", columns="qid", values="correct", aggfunc="mean")
    acc = acc.reindex(columns=[q for q in QIDS if q in acc.columns])
    first = s[s.qid == acc.columns[0]]
    n = first.groupby("subfolder").size()
    lab = first.groupby("subfolder")["emotion"].agg(lambda x: "/".join(sorted(x.unique())))
    acc.index = [f"{i} (n={n[i]}, {lab[i]})" for i in acc.index]
    return acc


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _annotate(ax, mat, fmt="{:.2f}", dark_above=None, dark_below=None, fontsize=8):
    for (i, j), v in np.ndenumerate(np.asarray(mat, dtype=float)):
        if np.isnan(v):
            continue
        white = (dark_above is not None and v > dark_above) or (dark_below is not None and v < dark_below)
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                color="white" if white else "black")


def plot_metric_bars(summary, metric="accuracy", lo="acc_lo", hi="acc_hi", ax=None):
    ax = ax or plt.gca()
    models = list(summary["model"].unique())
    qids = [q for q in summary["qid"].unique()]
    w = 0.8 / len(models)
    x = np.arange(len(qids))
    for i, m in enumerate(models):
        s = summary[summary.model == m].set_index("qid").reindex(qids)
        ax.bar(x + i * w - 0.4 + w / 2, s[metric], w, label=m,
               yerr=[s[metric] - s[lo], s[hi] - s[metric]], capsize=2)
    ax.set_xticks(x, [qlabel(q) if q in QMETA.index else q for q in qids], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=.3)
    if len(models) > 1:
        ax.legend(fontsize=7)
    return ax


def plot_heatmap(mat, title, ax, cmap="RdYlGn", vmin=0, vmax=1, xlabels=None, fontsize=8):
    ax.imshow(np.asarray(mat, dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    _annotate(ax, mat, fontsize=fontsize, dark_below=0.6 if cmap == "viridis" else None)
    ax.set_xticks(range(mat.shape[1]), xlabels or list(mat.columns), fontsize=7)
    ax.set_yticks(range(mat.shape[0]), list(mat.index), fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_confusions(long, model, fname=None):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, q in zip(axes.flat, QIDS):
        g = long[(long.model == model) & (long.qid == q)]
        cats = POLARITY_CATS if QMETA.loc[q, "target"] == "polarity" else YESNO_CATS
        cols = cats + [c for c in ["unparsed", "missing"] if (g.pred == c).any()]
        cm = pd.crosstab(g["emotion"], g["pred"]).reindex(index=LABEL_ORDER, columns=cols, fill_value=0)
        rel = cm.div(cm.sum(1).replace(0, 1), axis=0)
        ax.imshow(rel, cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm.to_numpy()):
            ax.text(j, i, v, ha="center", va="center", color="white" if rel.iat[i, j] > .5 else "black")
        ax.set_xticks(range(len(cols)), cols, rotation=45, fontsize=8)
        ax.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER, fontsize=8)
        ax.set_title(f"{q} ({QMETA.loc[q, 'target']}, {QMETA.loc[q, 'style']})", fontsize=9)
    fig.suptitle(f"Confusion matrices – {model} (rows = label, colour = row share)")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    return fig


AGREEMENT_GROUPS = {"polarity (incl. derived)": POL_COLS, "negative?": ["Q3", "Q4", "Q5"],
                    "positive?": ["Q6", "Q7", "Q8"]}


def plot_agreement(df, model, fname=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for k, (gname, cols) in enumerate(AGREEMENT_GROUPS.items()):
        agree, kap = pairwise_agreement(df, cols, model)
        plot_heatmap(agree, f"{gname}: % agreement", axes[0, k], cmap="viridis")
        plot_heatmap(kap, f"{gname}: Cohen's κ", axes[1, k], cmap="viridis")
    fig.suptitle(f"Agreement between questions – {model}")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=150)
    return fig


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
