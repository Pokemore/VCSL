#!/bin/sh

CUDA_VISIBLE_DEVICES='4,5,6,7' python -m torch.distributed.launch --nproc_per_node=4 --master_port 29500 --use_env main.py --use_sfd --sfd_atoms 512 --sfd_alpha 0.8 --dataset_file rsvg --binary --with_box_refine --batch_size 2 --num_frames 1 --epochs 70 --lr_drop 40 --num_queries 10 --output_dir rsvg_dirs/r50_bidrection_fusion_10query_70epo_multiscale --backbone resnet50

