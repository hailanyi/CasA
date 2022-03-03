#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=2 test.py --launcher pytorch > log-test.txt &

