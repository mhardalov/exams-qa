# EXAMS: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering

EXAMS is a new benchmark dataset for cross-lingual and multilingual question answering for high school examinations. 
It contains more than *24,000* high-quality high school exam questions in *26* languages, 
covering *8* language families and *24* school subjects from Natural Sciences and Social Sciences, among others. 
EXAMS offers a fine-grained evaluation framework across multiple languages and subjects, which allows precise 
analysis and comparison of various models. 

This repository contains links to the data, the models, and a set of scripts for preparing the dataset, and evaluating new models.

For more details on how the dataset was created, and baseline models testing multilingual and cross-lingual transfer,
please refer to our paper, [EXAMS: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual 
Question Answering](http://arxiv.org/abs/)

## Dataset

The data can be downloaded from here: (1) [Multilingual](https://github.com/mhardalov/exams-qa/data/exams/multilingual), 
(2) [Cross-lingual](https://github.com/mhardalov/exams-qa/data/exams/cross-lingual)

The two testbeds are described in the [paper](http://arxiv.org/abs/).
The files are in `jsonl` format and follow the [ARC Dataset's](https://allenai.org/data/arc) structure.
Each file is named using the following pattern: `data/exams/{testbed}/{subset}.jsonl`

We also provide the questions with the resolved contexts from Wikipedia articles. The files are in the `with_paragraphs` folders, 
folder, and are named `{subset}_with_para.jsonl`.

### Multilingual

In this setup, we want to train and to evaluate a given model with multiple languages, and thus we need multilingual 
training, validation and test sets. 
In order to ensure that we include as many of the languages as possible, we first split the questions independently 
for each language *L* into *Train<sub>L</sub>*, *Dev<sub>L</sub>*, *Test<sub>L</sub>* with 37.5%, 12.5%, 50% of the examples, respectively.

*For languages with fewer than 900 examples, we only have *Test<sub>L</sub>*.

| Language   | Train | Dev | Test |
| :--- | :---: | :---: |  :---: | 
| Albanian   | 565 | 185 | 755        |
| Arabic     | - | - | 562            |
| Bulgarian  | 1,100 | 365 | 1,472    |
| Croatian   | 1,003 | 335 | 1,541    |
| French     | - | - | 318            |
| German     | - | - | 577            |
| Hungarian  | 707 | 263 | 1,297      |
| Italian    | 464 | 156 | 636        |
| Lithuanian | - | - | 593            |
| Macedonian | 778 | 265 | 1,032      |
| Polish     | 739 | 246 | 986        |
| Portuguese | 346 | 115 | 463        |
| Serbian    | 596 | 197 | 844        |
| Spanish    | - | - | 235            |
| Turkish    | 747 | 240 | 977        |
| Vietnamese | 916 | 305 | 1,222      |
| Combined   | 7,961 | 2,672 | 13,510 |

### Cross-lingual

In this setting, we want to explore the capability of a model to transfer its knowledge from a single source 
language *L<sub>src</sub>* to a new unseen target language *L<sub>tgt</sub>*. In order to ensure that we have a larger 
training set, we train the model on 80% of *L<sub>src</sub>*, we validate on 20% of the same language, and we test on a 
subset of *L<sub>tgt</sub>*.

For this setup, we offer per-language subsets for both the train, and dev sets. The file naming patter is 
`{subset}_{lang}.jsonl}`, e.g., `train_ar.jsonl`, `train_ar_with_para.jsonl`, `dev_bg.jsonl`, etc.

Finally, in this setup the `test.jsonl` is the same one as in the `Multilignaul` one.

| Language   | Train | Dev | 
| :--- | :---: | :---: | 
| Albanian   | 1,194 | 311 |
| Arabic     | - | - |
| Bulgarian  | 2,344 | 593 |
| Croatian   | 2,341 | 538 |
| French     | - | - |
| German     | - | - |
| Hungarian  | 1,731 | 536 |
| Italian    | 1,010 | 246 |
| Lithuanian | - | - |
| Macedonian | 1,665 | 410 |
| Polish     | 1,577 | 394 |
| Portuguese | 740 | 184 |
| Serbian    | 1,323 | 314 |
| Spanish    | - | - |
| Turkish    | 1,571 | 393 |
| Vietnamese | 1,955 | 488 |

### Parallel Questions

The EXAMS dataset contains `10,000` paralell questions, therefore we also provide the mappings between questions 
in [jsonl format](https://github.com/mhardalov/exams-qa/data/exams/parallel_questions.jsonl). 
Each row from the file contains a mapping from `question id` to a list of parallel ones in other languages.

## Training and Evaluation

For both scripts the supported values for (multilingual) model types (`$MODEL_TYPE`) are: 
`"bert", "xlm-roberta", "bert-kb", "xlm-roberta-kb"`.

The paragraph type (`$PARA_TYPE`) modes are: `'per_choice', 'concat_choices', 'ignore'`

When using EXAMS with `run_multiple_choice` one should use `--task_name exams`, otherwise the one suitable for the task,
 e.g., `arc`, or `race`.

### Training

We use [HuggingFace's](https://github.com/huggingface/transformers) scripts for training the models, with slight modifications to allow for 3- to 5-way 
multiple-choice questions. The python scripts are available under the
[scripts/experiments](https://github.com/mhardalov/exams-qa/scripts/experiments) folder.

Here is an example:

```shell
python ./scripts/experiments/run_multiple_choice.py \
    --model_type $MODEL_TYPE \
    --task_name $TASK_NAME \
    --tb_log_dir runs/${TRAIN_OUTPUT_SUBDIR}/$RUN_SETTING_NAME \
    --model_name_or_path $TRAINED_MODEL_DIR \
    --do_train \
    --do_eval \
    --warmup_proportion ${WARM_UP} \
    --evaluate_during_training \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${LOGGING_STEPS} \
    --data_dir $TRAIN_DATA_DIR \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $MAX_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $TRAIN_OUTPUT \
    --weight_decay $WEIGHT_DECAY \
    --overwrite_cache \
    --per_gpu_eval_batch_size=$EVAL_BATCH_SIZE \
    --per_gpu_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --overwrite_output 
```

### Evaluation 

We provide an evaluation script that allows fine-grained evaluation on both subject, and language level. The script 
is available at [scripts/evaluation/evaluate_exams.py](https://github.com/mhardalov/exams-qa/data/exams/multilingual/scripts/evaluation/evaluate_exams.py).

Example usage:

```bash
python evaluate_exams.py --predictions_path predictions.json --dataset_path dev.jsonl --granularity all --output_path results.json
```


The possible granularities that the scripts supports are: `language`, `subject`, `subject_and_language`, and `all` 
(includes all other options). 

A sample predictions file can be found here: 
[sample_predictions.jsonl](https://github.com/mhardalov/exams-qa/scripts/evaluation/sample_predictions.jsonl).

### Predictions

The following script can be used to obtain predictions from pre-trained models.

```
python ./scripts/experiments/run_multiple_choice.py \
    --model_type $MODEL_TYPE \
    --task_name exams \
    --do_test \
    --para_type per_choice \
    --model_name_or_path $TRAINED_MODEL_DIR \
    --data_dir $INPUT_DATA_DIR \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --per_gpu_eval_batch_size=$EVAL_BATCH_SIZE \
    --overwrite_cache \
    --overwrite_output
```


## Contexts

The scripts used for downloading the Wikipedia articles, and context resolution can be in the 
[scripts/dataset](https://github.com/mhardalov/exams-qa/scripts/dataset) folder.

## Baselines

The EXAMS [paper](http://arxiv.org/abs/) presents several baselines for zero-shot, and few-shot training using publicly 
avaible multiple-choice datasets: RACE, ARC, OpenBookQA, Regents. 


### Multilingual

* The *(Full)* models are trained on all aforementioned datasets, including EXAMS.

|  Lang/Set |ar | bg | de | es | fr | hr | hu | it | lt | mk | pl | pt | sq | sr | tr | vi | All | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Random Guess | 25.0 | 25.0 | 29.4 | 32.0 | 29.4 | 26.7 | 27.7 | 26.0 | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 | 26.2 | 23.1 | 25.0 | 25.9 | 
|  IR (Wikipedia) | 31.0 | 29.6 | 29.3 | 27.2 | 32.1 | 31.9 | 29.7 | 27.6 | 29.8 | 32.2 | 29.2 | 27.5 | 25.3 | 31.8 | 28.5 | 27.5 | 29.5 | 
|  XLM-R on RACE | 39.1 | 43.9 | 37.2 | 40.0 | 37.4 | 38.8 | 39.9 | 36.9 | 40.5 | 45.9 | 33.9 | 37.4 | 42.3 | 35.6 | 37.1 | 35.9 | 39.1 | 
|  w/ SciENs | 39.1 | 44.2 | 35.5 | 37.9 | 37.1 | 38.5 | 37.9 | 39.5 | 41.3 | 49.8 | 36.1 | 39.3 | 42.5 | 37.4 | 37.4 | 35.9 | 39.6 | 
|  then  on Eχαμs (Full) | 40.7 | 47.2 | 39.7 | 42.1 | 39.6 | 41.6 | 40.2 | 40.6 | 40.6 | 53.1 | 38.3 | 38.9 | 44.6 | 39.6 | 40.3 | 37.5 | 42.0 | 
|  XLM-R<sub>Base</sub> (Full) | 34.5 | 35.7 | 36.7 | 38.3 | 36.5 | 35.6 | 33.3 | 33.3 | 33.2 | 41.4 | 30.8 | 29.8 | 33.5 | 32.3 | 30.4 | 32.1 | 34.1 | 
|  mBERT  (Full)  | 34.5 | 39.5 | 35.3 | 40.9 | 34.9 | 35.3 | 32.7 | 36.0 | 34.4 | 42.1 | 30.0 | 29.8 | 30.9 | 34.3 | 31.8 | 31.7 | 34.6 | 


## References

Coming soon...

## License

The dataset, which contains paragraphs from Wikipedia, is licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode). The code in this repository is licenced according the LICENSE file. 
