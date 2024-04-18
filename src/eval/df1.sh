#!/bin/sh

mode=$1
model=$2
cuda=$3

python to_roberta.py "$mode" "$model"

CUDA_VISIBLE_DEVICES="$cuda" python transformers/examples/pytorch/question-answering/run_qa.py \
--model_name_or_path ../../models/roberta-base-squad2 \
--validation_file ../../exp/eval/df1/test_m"$mode"_"$model"_qa.json \
--do_eval \
--version_2_with_negative \
--max_seq_length 384 \
--output_dir ../../exp/eval/df1/test_m"$mode"_"$model" \
--null_score_diff_threshold 0