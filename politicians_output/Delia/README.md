# Politician model results

Put the VQA result files here:

| file | used by |
|---|---|
| `data_out_politicians_Qwen-VL-3B.csv` (`filename`, `caption`, `vqa`) | `notebooks/question_accuracy_single_model.ipynb` |
| `comparison_vqa_politicians_OpenRouter.csv` (`filename` + one column per model) | `notebooks/question_accuracy_models.ipynb` |

Each `vqa` cell is a list with the answers to the 4 questions in `../list_of_questions17NOV26.txt`.
The `filename` column must keep the original image path (`.../politicians/putin trump zielinsky/<name>/<image>`), because the expected answer to Q4 (Trump, Zelenskyy or Putin) is derived from it.
