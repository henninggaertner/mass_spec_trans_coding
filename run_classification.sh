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
./scripts/quick-submit.sh -- python -m experiments.run_classification \
--encoded-directory /data/resnet_v2_101 \
--expression-directory /data/expression \
--output-directory /output/results/run_classification \
--index-csv /data/index.csv \
--annotation-csv /data/annotation.csv \
--patient-mapping /data/inline-supplementary-material-5.xlsx \
--cohort-identifier ppp1_raw_image_512x512 \
--n-jobs 16