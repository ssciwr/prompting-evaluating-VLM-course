# Compact Course: Prompting and Evaluating Multimodal AI Models with AMMICO

- Date: 20.11.2026
- Time: 9am - 1pm
- Instructor: PD Dr. Delia Dumitrescu, Political Sciences, IPW / Dr. Inga Ulusoy, Dmitrii Kapitan, Research Software Engineers, Scientific Software Center
- Venue: Mathematikon Bauteil A, Im Neuenheimer Feld 205, seminar room 12 on the 5th floor

This is a half-day course.

## Prerequisites

Participants should have basic familiarity with handling structured data such as CSV files. In addition, basic familiarity with Python and Jupyter notebooks, as well as with generative AI and large language models, is useful but not required. Participants are required to bring their own laptops to work on during the course. They need to have a registered ORCID account to be able to use the Jupyter4NFDI service.

## Summary
Multimodal AI models can analyze text, images, video, and combinations thereof, enabling researchers to extract structured information from increasingly large and heterogeneous datasets. However, the results of these models depend strongly on the selected model and model parameters, the dataset, and - particularly - the way a task is formulated through prompts.
In this hands-on course, participants will use AMMICO, an open-source Python package for AI-based media and content analysis, to systematically explore these factors. We will apply vision-language and audio models to different datasets and compare different model backends for tasks such as content description, feature extraction, and visual question answering.

A particular focus will be on prompt design. Participants will experiment with different ways of formulating the same research question and investigate how prompt specificity, context, output constraints, and examples influence model responses. We will develop prompts iteratively and discuss strategies for obtaining more consistent, interpretable, and reusable results.
The course will discuss practical approaches for evaluating outputs and documenting model and prompt configurations to make AI-assisted analyses more transparent and reproducible.

## Learning Objectives

After the course participants will be able to:
- use AMMICO to perform AI-based analysis of image and multimodal datasets;
- formulate research questions as prompts for multimodal AI models;
- apply prompt-design strategies such as providing context, defining explicit tasks, specifying output formats, and using examples;
- iteratively improve prompts based on observed model outputs;
- compare the behavior of different model backends and datasets for the same analytical task;
- evaluate the consistency and suitability of model outputs for research questions;
- recognize limitations and sources of variability in generative AI-based analysis; and
- document datasets, models, parameters, and prompts to support transparent and reproducible research workflows. 

## Datasets

The course uses three image datasets. Each comes with a ground truth (annotated by hand) and a list of questions (prompts) that ask for the same information in different ways:

| dataset | images | ground truth | questions |
|---|---|---|---|
| Politicians: is there a politician in the picture? | `dataset_politicians/` (255) | [`politicians_labels.csv`](politicians_output/politicians_labels.csv) | [4 questions](politicians_output/politicians_questions17NOV26.txt) |
| Emotions: does the picture show positive or negative emotional content? | `dataset_emotions/` (300) | [`emotions_labels.csv`](emotions_output/emotions_labels.csv) | [8 questions](emotions_output/emotions_questions16NOV26.txt) |
| Symbols: does the picture show a national or EU symbol? | `dataset_symbols/` (141) | [`symbol_labels.csv`](symbols_output/symbol_labels.csv) | [4 questions](symbols_output/symbols_questions16NOV26.txt) |

The images are not part of this repository. In the course they are provided in the workshop folder, together with copies of the notebooks and the ground truth:

```
workshop_112026/
├── dataset_emotions/        images (one flat folder per dataset)
├── dataset_politicians/
├── dataset_symbols/
├── ground_truth/            label csv files and question files (as in the *_output/ folders here)
└── comparison_notebooks/    the notebooks from notebooks/
```

## Workflow

1. **Generate the answers.** Run the visual question answering with AMMICO on the images of one dataset, using the questions of that dataset (all of them, in the same order). Save the result as a csv file, e.g. `my_results/emotions_vqa.csv`. Repeat with other models to compare models.
2. **Evaluate the questions.** Open one of the notebooks in [`notebooks/`](notebooks/) and set the two paths in the SETTINGS cell: the result file(s) and the image folder.
   - [`question_accuracy_single_model.ipynb`](notebooks/question_accuracy_single_model.ipynb): which question gets the most images right for one model?
   - [`question_accuracy_models.ipynb`](notebooks/question_accuracy_models.ipynb): the same for several models, and do the models agree on the best question?

The dataset is detected from the image names in the result file, and the ground truth, questions and expected answers are taken from `DATASETS` in [`notebooks/vqa_eval_utils.py`](notebooks/vqa_eval_utils.py). The ground-truth files are read from `ground_truth/` next to the image folder, or from the `*_output/` folders of this repository. The result file needs a column `filename` (the image name or path) and a column `vqa` with the list of answers per image (the AMMICO output); a column `model` is optional, otherwise the file name is used as the model name. Tables and plots are saved in `comparison_results/` next to the result file. The images are only needed for the image examples at the end of the notebooks.

## Politicians dataset

Images with and without politicians. Q1–Q3 ask whether there is a politician in the picture (open, constrained to yes / no / dk, constrained with a justification); they are scored against the column `politician`. Q4 asks whether Donald Trump, Volodymyr Zelenskyy or Vladimir Putin is in the picture; it is scored against the column `trump_zelenskyy_putin` and is not part of the summary, as it asks a different question.

## Emotions dataset

Images with negative, neutral and positive emotional content (column `emotion`), coming from different sources (column `subfolder`, e.g. neutral visuals with negative text). The 8 questions ask for the polarity (positive / negative / both / none), for negative content or for positive content, each in an open and a constrained form, and with cues where the emotion can come from (faces and bodies, what is depicted, the text).

## Symbols dataset

Images with and without national or EU symbols (column `human_nat_eu_symbol`, grouped into EU flag, other national or EU symbol, no symbol in `human_category`). The 4 questions differ in the examples they give and whether they point out symbols that are only partly visible or blended into the image.
