# Symbol model results

Put these files from the `results/` folder of the national/EU symbols reproducible package here:

| file | content |
|---|---|
| `merged_coded_symbols.csv` | raw answers (`<model>__q<i>_raw`) and coded answers (`<model>__q<i>_code`: 1 = yes, 0 = no, 9 = DK, 8 = unclear) for all models and questions |
| `prompts.csv` | the 4 questions |
| `model_key.csv` | model names and their column keys |

`notebooks/question_accuracy_models.ipynb` compares all models in the file.
