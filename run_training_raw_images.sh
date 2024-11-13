#!/bin/bash
# the following environment variables will be used by the quick-submit.sh script, and deeper down the submit.sh script.
export JOB_NAME="train"
export NUM_CPU=32
export MEMORY="30G"
export NUM_GPU=2
export PARTITION="sorcery"
export ACCOUNT="renard"
export TIME="2:0:0"
export CONSTRAINTS="ARCH:X86"
./scripts/quick-submit.sh -- python -m experiments.run_training_raw_images \
--input-directory /data/ppp1_raw_image_512x512 \
--output-directory /output/results/run_training_raw_images \
--batch-size 32 \
--index-csv /data/index.csv \
--annotation-csv /data/annotation.csv \
--patient-mapping /data/inline-supplementary-material-5.xlsx