## AmbigDocs: Reasoning across Documents on Different Entities under the Same Name

[[Paper]](./) [[Homepage]](https://ambigdocs.github.io) [[Dataset]](https://huggingface.co/datasets/yoonsanglee/AmbigDocs/tree/main)

### Introduction
This is the repository for the paper AmbigDocs: Reasoning across Documents on Different Entities under the Same Name.

We introduce AmbigDocs, a benchmark for testing the abilities of current LMs to distinguish confusing entity mentions and generate a cohesive answer. Single instance consists of a question asking about an ambiguous entity and a list of gold document-answer pairs for each disambiguated entity.

### Dataset Contents
Download the data from [here](https://huggingface.co/datasets/yoonsanglee/AmbigDocs/tree/main) and place under [data](./data). Additionally, we use the Wikipedia snapshot from December 20th, 2018. Please place the documents (psgs_w100.tsv) in same directory, which can be downloaded from [DPR repo](https://github.com/facebookresearch/DPR).

Each data instance consists of `question`, `ambiguous_entity`, `qid`, and a list of `documents`. Each element in `documents` consists of `title` which is a disambiguated entity, `text`, `pid` for referencing psgs_w100.tsv, and `answer`.

### Setup
```
pip install -r requirements.txt
```

For evaluation, please place the necessary LMs under [models](./models). For generation, please place [question_converter-3b](https://huggingface.co/domenicrosati/question_converter-3b), [t5_xxl_true_nli_mixture](https://huggingface.co/google/t5_xxl_true_nli_mixture) under [models](./models).

### Dataset Generation
For dataset generation, please refer to [generation](./generation) subdirectory.

### Evaluation
1. Executing below will run inference on test split. `mode` represents the following: 
`1: Gold Only, 2: Gold+Retrieved, 3: Retrieved Only, 4: Few-shot` Put the name of the model you are using in `model`. If this contains "gpt", put openAPI key afterwards. Otherwise, put the model path to the argument.
    ```
    python qa.py [mode] [model] [openAPI key/path_to_QA_model]
    ``` 

2. Executing below will compute preliminary operations for computing Disambig-F1 score.
    ```
    sh df1.sh [mode] [model]
    ``` 

3. Executing below will compute Answer Recall / Entity Recall / Entity-Answer Recall / Disambig-F1 scores.
    ```
    python eval.py [mode] [model]
    ``` 

While our study mainly focuses on `Gold Only` setting, we also experiment on retrieved corpus. We leverage GTR as our retriever and the codes taken from [ALCE repo](https://github.com/princeton-nlp/ALCE). Please download necessary pre-computed embeddings and GTR model and execute [retrieval.py](./src/eval/ALCE/retrieval.py).

```
python retrieval.py [path_to_gtr_wikipedia_index.pkl] [path_to_GTR_model] \
--retriever gtr \
--data_file ../../../data/test.json \
--output_file ../../../data/test_retrieved.json
```

### Citations
If you find our work helpful, please cite us as
```
@article{ lee2024ambigdocs }
```