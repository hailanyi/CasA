#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5 nohup python3 -m torch.distributed.launch --nproc_per_node=5 train.py --launcher pytorch > log.txt&
