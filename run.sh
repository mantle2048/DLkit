#!/usr/bin/env bash

algos_dir='DLkit/algos/vae/vae.py'

python $algos_dir \
    --data_dir 'dataset' \
    --use_gpu \
    --seed 2 \
    --epoches 20 \
    --datestamp \
    --learning_rate 1e-3 \
    --which_gpu 0
