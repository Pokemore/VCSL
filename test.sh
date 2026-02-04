#!/bin/sh

python3 inference_rsvg.py --use_sfd --sfd_atoms 512 --sfd_alpha 0.8 --dataset_file rsvg --num_queries 10 --with_box_refine --binary --freeze_text_encoder --resume rsvg_dirs/r50_bidrection_fusion_10query_70epo/checkpoint.pth --backbone resnet50

