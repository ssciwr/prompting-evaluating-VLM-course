"""Shared helpers for comparing annotated labels with VLM answers.

Used by the question-accuracy notebooks of the emotions dataset (emotions_output/) and the
politicians dataset (politicians_output/). The emotion setup is the default; another dataset
calls ``set_questions(load_questions(path, targets=..., styles=...))`` first.

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


def load_questions(path, targets=TARGETS, styles=STYLES, label_cols=None):
    """Read a question file (a python list assignment) into the question table.

    targets    : what each question asks about; selects the normalisation rules and the truth mapping
                 (emotions: polarity / negative / positive; politicians: politician / named)
    styles     : how each question is worded (open / constrained / ...), only used for labels in plots
    label_cols : label column the expected answer is derived from (default: "label" for every question)
    """
    txt = Path(path).read_text()
    questions = ast.literal_eval(txt[txt.index("["):txt.rindex("]") + 1])
    assert len(questions) == len(targets) == len(styles), \
        f"{len(questions)} questions but {len(targets)} targets / {len(styles)} styles"
    qids = [f"Q{i}" for i in range(1, len(questions) + 1)]
    return pd.DataFrame({"qid": qids, "target": list(targets), "style": list(styles), "question": questions,
                         "label_col": list(label_cols) if label_cols else ["label"] * len(qids)}).set_index("qid")


QMETA = pd.DataFrame({"qid": QIDS, "target": TARGETS, "style": STYLES, "label_col": "label"}).set_index("qid")


def set_questions(qmeta):
    """Make `qmeta` the active question set for all functions of this module."""
    global QIDS, QMETA
    QMETA = qmeta
    QIDS = list(qmeta.index)
    return qmeta


SHORT = {"polarity": "pol", "negative": "neg", "positive": "pos", "politician": "politician", "named": "T/Z/P",
         "symbol": "symbol"}


def qlabel(q):
    return f"{q}\n{SHORT.get(QMETA.loc[q, 'target'], QMETA.loc[q, 'target'])}/{QMETA.loc[q, 'style']}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_labels(path, label_col="emotion"):
    """Label table with a generic `label` column (the original column is kept as well)."""
    lab = pd.read_csv(path)
    lab["filename"] = lab["filename"].str.strip()
    lab["label"] = lab[label_col].astype(str).str.strip()
    lab["subfolder"] = lab["subfolder"].fillna("(none)") if "subfolder" in lab.columns else "(none)"
    assert not lab["filename"].duplicated().any(), "duplicate filenames in labels"
    return lab


def parse_vqa_list(s, n=None):
    """Return a list of length n (default: number of questions); entries are None when the answer is missing/empty."""
    n = n or len(QIDS)
    try:
        lst = ast.literal_eval(s) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        lst = []
    lst = [a if isinstance(a, str) and a.strip() else None for a in lst]
    return (lst + [None] * n)[:n]


def load_vqa(path, model_name=None):
    """Wide table: filename, model, Q1..Q8 (raw answers, None = missing)."""
    raw = pd.read_csv(path)
    out = pd.DataFrame({"filename": raw["filename"].map(os.path.basename), "source_path": raw["filename"]})
    if "model" in raw.columns and model_name is None:
        out["model"] = raw["model"].values
    else:
        out["model"] = model_name or Path(path).stem
    answers = raw["vqa"].map(parse_vqa_list)
    for i, q in enumerate(QIDS):
        out[q] = answers.map(lambda l, i=i: l[i])
    assert not out.duplicated(["model", "filename"]).any(), f"duplicate filenames in {path}"
    return out


def load_vqa_wide(path, model_columns=None):
    """Wide result file with one column per model (each cell = list of answers) -> same format as load_vqa."""
    raw = pd.read_csv(path)
    model_columns = model_columns or [c for c in raw.columns if c != "filename"]
    frames = []
    for m in model_columns:
        out = pd.DataFrame({"filename": raw["filename"].map(os.path.basename), "source_path": raw["filename"], "model": m})
        answers = raw[m].map(parse_vqa_list)
        for i, q in enumerate(QIDS):
            out[q] = answers.map(lambda l, i=i: l[i])
        frames.append(out)
    out = pd.concat(frames, ignore_index=True)
    assert not out.duplicated(["model", "filename"]).any(), f"duplicate filenames in {path}"
    return out


def load_coded_wide(path, model_keys):
    """Pre-coded result table (symbols package: columns <key>__q<i>_raw and <key>__q<i>_code).

    model_keys: {column key: model name}. Returns the load_vqa format plus Qx_code columns, which
    `normalise` uses instead of parsing the raw text (codes: 1 = yes, 0 = no, 9 = dk, 8 = unclear).
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    frames = []
    for key, name in model_keys.items():
        out = pd.DataFrame({"filename": raw["filename"].map(os.path.basename), "model": name})
        for i, q in enumerate(QIDS, start=1):
            out[q] = raw[f"{key}__q{i}_raw"].where(raw[f"{key}__q{i}_raw"].notna(), None)
            out[q + "_code"] = raw[f"{key}__q{i}_code"]
        frames.append(out)
    out = pd.concat(frames, ignore_index=True)
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


def normalise(df, qmeta=None):
    """Add Qx_norm and Qx_rule columns."""
    qmeta = QMETA if qmeta is None else qmeta
    out = df.copy()
    for q in qmeta.index:
        if q + "_code" in out.columns:  # answers already coded (e.g. symbols package incl. human review)
            out[q + "_norm"] = out[q + "_code"].map(CODE_MAP).fillna("unparsed")
            out[q + "_rule"] = "package_code"
            continue
        res = out[q].map(NORMALISERS[qmeta.loc[q, "target"]])
        out[q + "_norm"] = res.map(lambda r: r[0])
        out[q + "_rule"] = res.map(lambda r: r[1])
    return out


# ---------------------------------------------------------------------------
# Politicians dataset: "Is there a politician in the picture?" (target 'politician') and
# "Is any of Trump / Zelenskyy / Putin in the picture?" (target 'named')
# ---------------------------------------------------------------------------
POLITICIAN_WORDS = (r"politicians?|political (figure|leader)s?|president|prime minister|minister|chancellor|senator"
                    r"|member of (the )?(european )?parliament|\bmp\b|mep|mayor|head of (state|government)"
                    r"|statesman|congress(wo)?man|governor|party leader|lawmaker")
NON_POLITICIAN_WORDS = (r"coach|footballer|football (player|manager)|player|athlete|actor|actress|singer|musician"
                        r"|journalist|news anchor|anchor|presenter|reporter|officer|soldier|celebrity|businessman"
                        r"|businesswoman|entrepreneur|director|artist|model|influencer|doctor|scientist")
POL_DK = re.compile(
    r"^i (don'?t|do not) know|i (can'?t|cannot|am unable to|'m unable to|am not able to) (identify|determine|tell|say|recogni)"
    r"|(not|isn'?t) possible to (definitively |reliably |clearly |accurately )?(determine|identify|tell|say)|cannot (definitively |reliably |accurately )?(say|determine|identify|tell)"
    r"|not (clear|enough|sufficient)[^.]{0,30}(identify|determine)|does not provide (enough|sufficient)"
    r"|difficult to (determine|identify|say|tell)|unable to (determine|identify)|without (more |additional )?(context|information)")
NAMED_WORDS = r"trump|zelens\w*|zelenskyy|selenskyj|putin"


def _norm_yes_no_text(s, own_words, other_words, dk_re, none_of_listed=False):
    if _isna(s):
        return "missing", "missing"
    c = _clean(s)
    if c in YESNO_CATS:
        return c, "exact"
    if none_of_listed and re.search(r"none of the (mentioned|listed|following|named|three)", c):
        return "no", "kw_none_of_listed"
    w = _first_word(c)
    if w in YESNO_CATS:
        return w, "first"
    a = _explicit_answer(c)
    if a in YESNO_CATS:
        return a, "answer_is"
    if re.search(r"\bhighly unlikely\b|\bunlikely that there is\b", c):
        return "no", "kw_negated"
    if dk_re.search(c):  # hedging first: "it is not possible to determine if there is a politician" = dk, not no
        return "dk", "kw_dk"
    if re.search(rf"{NEGATOR}[^.;]{{0,40}}?\b({own_words})\b", c):
        return "no", "kw_negated"
    if re.search(rf"\b({own_words})\b", c):
        return "yes", "kw_affirm"
    if other_words and re.search(rf"\b({other_words})\b", c):
        return "no", "kw_opposite"
    return "unparsed", "unparsed"


def norm_politician(s):
    """'Is there a politician in the picture?' -> yes | no | dk | unparsed | missing."""
    return _norm_yes_no_text(s, POLITICIAN_WORDS, NON_POLITICIAN_WORDS, POL_DK)


def norm_named(s):
    """'Is Trump / Zelenskyy / Putin in the picture?' -> yes | no | dk | unparsed | missing."""
    return _norm_yes_no_text(s, NAMED_WORDS, None, POL_DK, none_of_listed=True)


# Symbols dataset: "Does the picture depict a national or EU symbol?" (target 'symbol')
SYMBOL_WORDS = r"flags?|emblems?|coat of arms|national symbols?|eu symbols?|european union symbols?"
CODE_MAP = {1: "yes", 0: "no", 9: "dk", 8: "unclear"}  # codes of the symbols reproducible package


def norm_symbol(s):
    """Fallback text rule for symbol questions (normally the package codes are used)."""
    return _norm_yes_no_text(s, SYMBOL_WORDS, None, POL_DK)


NORMALISERS = {
    "symbol": norm_symbol,
    "polarity": norm_polarity,
    "negative": lambda s: norm_yesno(s, "negative"),
    "positive": lambda s: norm_yesno(s, "positive"),
    "politician": norm_politician,
    "named": norm_named,
}


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
    "politician": {"yes": "yes", "no": "no"},
    "named": {"yes": "yes", "no": "no"},
    "symbol": {"1": "yes", "0": "no"},
}
LABEL_ORDER = ["negative", "neutral", "positive", "unclear"]


def _long_part(df, qid, target, style, pred_col, raw_col=None, label_col="label"):
    keep = ["model", "filename", "label", "subfolder"] + [c for c in ("emotion", "group") if c in df.columns]
    part = df[keep].copy()
    part["qid"], part["target"], part["style"] = qid, target, style
    part["raw"] = df[raw_col] if raw_col else None
    part["pred"] = df[pred_col]
    part["truth"] = df[label_col].map(TRUTH[target])  # NaN for labels without an expected answer ('unclear')
    part["scored"] = part["truth"].notna() & (part["pred"] != "missing")
    part["correct"] = np.where(part["scored"], part["pred"] == part["truth"], np.nan)
    return part


def to_long(df, qmeta=None):
    """One row per model x image x question."""
    qmeta = QMETA if qmeta is None else qmeta
    return pd.concat([_long_part(df, q, qmeta.loc[q, "target"], qmeta.loc[q, "style"], q + "_norm", q,
                                 qmeta.loc[q, "label_col"] if "label_col" in qmeta.columns else "label")
                      for q in qmeta.index], ignore_index=True)


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


# ---------------------------------------------------------------------------
# Simple question-accuracy analysis (one criterion: accuracy, DK = wrong)
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def accuracy_table(long, by=("model", "qid")):
    """n, correct, accuracy with Wilson CI, and DK share for each group (scored rows only)."""
    rows = []
    for key, g in long[long.scored].groupby(list(by), sort=False):
        n, k = len(g), int(g.correct.sum())
        lo, hi = wilson_ci(k, n)
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update(n=n, correct=k, accuracy=k / n, ci_low=lo, ci_high=hi, dk_share=float((g.pred == "dk").mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def pairwise_question_tests(long, model):
    """Exact McNemar test for every pair of questions of one model (same images), Holm-corrected."""
    w = long[(long.model == model) & long.scored].pivot(index="filename", columns="qid", values="correct")
    qids = [q for q in QIDS if q in w.columns]
    rows = []
    for i, a in enumerate(qids):
        for b in qids[i + 1:]:
            d = w[[a, b]].dropna().astype(bool)
            only_a, only_b = int((d[a] & ~d[b]).sum()), int((~d[a] & d[b]).sum())
            rows.append({"model": model, "q_a": a, "q_b": b, "n": len(d), "acc_a": d[a].mean(), "acc_b": d[b].mean(),
                         "only_a_correct": only_a, "only_b_correct": only_b, "p_value": mcnemar_exact(only_a, only_b)})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_value)
    t["significant"] = t.p_holm < 0.05
    return t


def plot_pairwise(tests, ax, title=""):
    """Heatmap of accuracy differences (row question minus column question); * = significant after Holm."""
    mat = pd.DataFrame(np.nan, index=QIDS, columns=QIDS)
    sig = pd.DataFrame(False, index=QIDS, columns=QIDS)
    for r in tests.itertuples():
        mat.loc[r.q_a, r.q_b], mat.loc[r.q_b, r.q_a] = r.acc_a - r.acc_b, r.acc_b - r.acc_a
        sig.loc[r.q_a, r.q_b] = sig.loc[r.q_b, r.q_a] = r.significant
    lim = max(0.1, np.nanmax(np.abs(mat.to_numpy())))
    ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu", vmin=-lim, vmax=lim)
    for (i, j), v in np.ndenumerate(mat.to_numpy(dtype=float)):
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.2f}" + ("*" if sig.iat[i, j] else ""), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(QIDS)), QIDS)
    ax.set_yticks(range(len(QIDS)), [qlabel(q).replace("\n", " ") for q in QIDS], fontsize=8)
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Difficult images
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def image_index(*image_dirs):
    """Map file name -> path for all images below the given folders (the first folder that exists wins)."""
    for d in image_dirs:
        d = Path(d).expanduser()
        if d.is_dir():
            return {p.name: p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}
    return {}


def image_difficulty(long):
    """Per image: number of scored answers, number wrong and share wrong (over all models and questions in `long`)."""
    s = long[long.scored]
    keys = ["filename", "label", "subfolder"] + (["group"] if "group" in s.columns else [])
    d = (s.groupby(keys)["correct"]
         .agg(n_answers="count", n_correct="sum").reset_index())
    d["n_wrong"] = d.n_answers - d.n_correct
    d["share_wrong"] = d.n_wrong / d.n_answers
    return d.drop(columns="n_correct").sort_values(["share_wrong", "filename"], ascending=[False, True])


def difficulty_by_group(diff, by="subfolder", hard_threshold=0.5):
    g = diff.groupby(by)
    return pd.DataFrame({"images": g.size(), "mean_share_wrong": g.share_wrong.mean(),
                         f"hard_images (≥{hard_threshold:.0%} wrong)": g.share_wrong.apply(lambda x: int((x >= hard_threshold).sum()))}
                        ).sort_values("mean_share_wrong", ascending=False)


def pick_examples(diff, per_label=2, by="label", order=None):
    """The `per_label` hardest images of each value of `by` (e.g. each label)."""
    order = order or sorted(diff[by].unique())
    return pd.concat([diff[diff[by] == e].head(per_label) for e in order])


def _load_thumbnail(path, size=700):
    from PIL import Image
    img = Image.open(path)
    img.thumbnail((size, size))
    return img.convert("RGB")


def plot_image_evaluation(filename, long, images, title=""):
    """Image (left) and a grid of the evaluation (right): one row per model, one column per question.
    Cell = normalised answer; green = correct, red = wrong (DK counts as wrong), grey = missing / not scored."""
    g = long[long.filename == filename]
    models = list(dict.fromkeys(g.model))
    first = g.iloc[0]
    expected = g.drop_duplicates("qid").set_index("qid").truth.reindex(QIDS)

    fig, (ax_img, ax_tab) = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    path = images.get(filename)
    if path is not None:
        ax_img.imshow(_load_thumbnail(path))
    else:
        ax_img.text(0.5, 0.5, "image not found", ha="center", va="center")
    ax_img.axis("off")
    ax_img.set_title(f"{filename}\nlabel: {first.label}  |  {first.get('group', first.subfolder)}", fontsize=9)

    rows = ["expected"] + models
    color = np.full((len(rows), len(QIDS)), 0.5)  # 0 = wrong, 1 = correct, 0.5 = neutral
    text = [[("" if pd.isna(expected[q]) else expected[q]) for q in QIDS]]
    for i, m in enumerate(models, start=1):
        r = g[g.model == m].set_index("qid").reindex(QIDS)
        text.append(list(r.pred.fillna("")))
        for j, q in enumerate(QIDS):
            if r.scored.get(q, False):
                color[i, j] = float(r.correct[q])
    from matplotlib.colors import ListedColormap
    ax_tab.imshow(color, cmap=ListedColormap(["#f4a6a6", "#dddddd", "#a8dba8"]), vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(QIDS)):
            ax_tab.text(j, i, text[i][j], ha="center", va="center", fontsize=8,
                        fontweight="bold" if i == 0 else "normal")
    ax_tab.set_xticks(range(len(QIDS)), [qlabel(q) for q in QIDS], fontsize=7)
    ax_tab.set_yticks(range(len(rows)), rows, fontsize=8)
    ax_tab.axhline(0.5, color="black", lw=1)
    n_s = g.scored.sum()
    ax_tab.set_title(f"{title}  wrong: {int(n_s - g.correct.sum())} of {int(n_s)} answers "
                     "(green = correct, red = wrong)", fontsize=9)
    fig.tight_layout()
    return fig


def wrong_answers(filename, long, max_chars=160):
    """Raw answers behind the wrong evaluations of one image."""
    g = long[(long.filename == filename) & long.scored & (long.correct == 0)]
    out = g[["model", "qid", "truth", "pred", "raw"]].copy()
    out["raw"] = out.raw.astype(str).str.replace("\n", " ").str.slice(0, max_chars)
    return out.reset_index(drop=True)
