#!/bin/bash
# the following environment variables will be used by the quick-submit.sh script, and deeper down the submit.sh script.
export JOB_NAME="train"
export NUM_CPU=16
export NUM_GPU=0
export MEMORY="30G"
export PARTITION="magic"
export ACCOUNT="renard"
export TIME="2:0:0"
export CONSTRAINTS="ARCH:X86"
./scripts/quick-submit.sh -- python experiments/run_classification.py \
--input_directory /data/ppp1_raw_image_512x512 \
--output_directory /data/results/run_classification \
--batch_size 4 \
--index_csv /data/index.csv \
--annotation_csv /data/annotation.csv \
--patient_mapping /data/inline-supplementary-material-5.xlsx