# Missing pieces for the question-accuracy notebooks

The notebooks in `notebooks/` (`question_accuracy_single_model.ipynb`, `question_accuracy_models.ipynb`) only need
two settings: the result csv file(s) and the image folder. The dataset (emotions, politicians, symbols) is detected
from the image names, and questions, ground truth and expected answers come from `DATASETS` in
`notebooks/vqa_eval_utils.py`. The following is still missing or out of date.

## 1. Data that is not in the repository

### Model result files
None of the result files are in the repo (removed with the raw data). The notebooks accept three formats, detected
automatically:

| format | columns | example |
|---|---|---|
| ammico output | `filename` + `vqa` (list with one answer per question), optional `model` | `emotions_data_out.csv`, `openai__gpt-4o-mini.csv` |
| one column per model | `filename` + one column per model (each cell a list of answers) | `comparison_vqa_politicians_OpenRouter.csv` |
| coded (symbols package) | `filename` + `<model>__q<i>_raw` / `<model>__q<i>_code` (1 = yes, 0 = no, 9 = DK, 8 = unclear) | `merged_coded_symbols.csv` |

- [ ] Decide where participants get the result files and where they put them; the notebooks take any path.
- [ ] Update the SETTINGS cells: they still point to the removed files
      `../emotions_output/emotions_data_out.csv` and `../emotions_output/<model>.csv`.
- [ ] Note for the symbols file: it contains several models, so it only works with
      `question_accuracy_models.ipynb` (the single-model notebook stops with a message).

### Images
No images are in the repo (`symbols_output/images/symbol_images/` has empty folders only). A **flat folder works**:
images are found by file name anywhere below `IMAGE_DIR`. They are only needed for the image examples in the last
section; all accuracy results work without them.

- [ ] Emotions images
- [ ] Politicians images
- [ ] Symbols images

## 2. Politicians: label for Q4 (Trump / Zelenskyy / Putin)

Q4 asks whether Trump, Zelenskyy or Putin is in the picture. Its expected answer is not in
`politicians_output/politicians_labels.csv`; it is derived from the folder
`politicians/putin trump zielinsky/` in the image path (in the result file or in `IMAGE_DIR`).
With flat folders **and** flat paths in the result file, every image gets "no" for Q4 and the group
"Trump/Zelenskyy/Putin" is empty (the notebook prints a warning).

- [ ] Add a column `trump_zelenskyy_putin` (`yes` / `no`) to `politicians_output/politicians_labels.csv`.
      If the column exists, it is used and the folder structure is not needed.

## 3. ~~Code that has to follow the renamed question files~~ (done)

- [x] The `questions=` paths in `DATASETS` point to the renamed files
      (`emotions_questions16NOV26.txt`, `politicians_questions17NOV26.txt`, `symbols_questions16NOV26.txt`).
- [x] `read_questions` also reads files without `[` `]` around the questions (as in `symbols_questions16NOV26.txt`).
      If a question file is renamed again, update its path in `DATASETS`; a missing file only shows placeholders
      instead of the question texts.

## 4. Documentation

- [ ] `README.md`: the emotions section links to `emotions_output/compare_emotions_data_out.ipynb` and
      `emotions_output/compare_emotions_models.ipynb`, which no longer exist; link to the notebooks in `notebooks/`.
- [ ] `README.md`: the politicians and symbols sections are empty.
- [ ] The READMEs that described where to put result files and images (`politicians_output/Delia/`,
      `symbols_output/results/`, `.../images/`, `emotions_output/emo_data/`) were removed; describe this
      somewhere for participants (e.g. in `README.md`).

## 5. Optional

- [ ] Symbols model names are the column keys of the coded file (e.g. `openai_gpt_4o_mini`). Nicer names could be
      read from the package's `model_key.csv`.
- [ ] There is no `.pre-commit-config.yaml` in the repo, so the code is not linted.
