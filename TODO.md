# Missing pieces for the question-accuracy notebooks

## Politicians: label for Q4 (Trump / Zelenskyy / Putin)

Q4 asks whether Trump, Zelenskyy or Putin is in the picture. Its expected answer is not in
`politicians_output/politicians_labels.csv`, and the images in `dataset_politicians/` are in one flat folder, so it
cannot be derived from the folder `politicians/putin trump zielinsky/` either. Until the label exists, Q4 is **not
scored** (the notebooks print a warning; Q1–Q3 are not affected).

- [ ] Add a column `trump_zelenskyy_putin` (`yes` / `no`) to `politicians_output/politicians_labels.csv` and to the
      copy in `workshop_112026/ground_truth/`.

## Optional

- [ ] There is no `.pre-commit-config.yaml` in the repo, so the code is not linted.
