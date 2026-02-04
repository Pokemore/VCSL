#!/usr/bin/env python3
"""
Simple one-batch forward + loss debug script for LQVG.

Run this on a machine with CUDA available and the project's deps installed.
It builds the model+criterion using the project's build() helper, loads one sample
from the RSVG dataset, runs a forward and prints output shapes and loss values.

Usage (from repo root):
  python -m LQVG.tools.debug_forward --use_rd

"""
import argparse
import torch
from pathlib import Path

from LQVG.models.LQVG import build as build_model
import LQVG.opts as opts
import LQVG.datasets.rsvg as rsvg


def move_target_to_device(target, device):
    for k, v in target.items():
        if isinstance(v, torch.Tensor):
            target[k] = v.to(device)
    return target


def main():
    parser = argparse.ArgumentParser("LQVG one-batch forward debug", parents=[opts.get_args_parser()])
    parser.add_argument('--image_set', default='train', choices=['train', 'val', 'test'])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("CUDA not available. This debug requires CUDA because deformable attention ops are CUDA-only.")
        return

    print(f"Building model (use_rd={args.use_rd}) on device {device}")
    model, criterion, _ = build_model(args)
    model.to(device)
    criterion.to(device)
    model.eval()

    print("Building dataset and loading one sample...")
    dataset = rsvg.build(args.image_set, args)
    sample, target = dataset[0]
    # sample: usually a Tensor [T, C, H, W]

    # move sample tensors to device
    def _move_to_device(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, (list, tuple)):
            return type(x)(_move_to_device(v, device) for v in x)
        return x

    sample = _move_to_device(sample, device)

    # prepare batch
    samples = [sample]  # batch size 1
    captions = [target.get('caption', "")]  # list of strings
    target = move_target_to_device(target, device)
    targets = [target]

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(samples, captions, targets)

    print("Outputs keys:", list(outputs.keys()))
    if 'pred_logits' in outputs and 'pred_boxes' in outputs:
        print('pred_logits.shape =', outputs['pred_logits'].shape)
        print('pred_boxes.shape  =', outputs['pred_boxes'].shape)

    # print targets and preds in human-readable form and compute IoU/GIoU
    try:
        from util import box_ops
        import torch.nn.functional as F

        # outputs: [batch, time, nq, 4]
        pred_boxes = outputs['pred_boxes']
        pred_logits = outputs['pred_logits']

        print('\nTarget boxes:')
        for i, t in enumerate(targets):
            print(f'  batch {i} boxes (cxcywh norm):', t['boxes'])

        # flatten over batch/time for metric computation
        b = pred_boxes.shape[0]
        t = pred_boxes.shape[1]
        nq = pred_boxes.shape[2]
        pred_boxes_flat = pred_boxes.view(b * t * nq, 4)

        # repeat targets to match shape if needed
        tgt_boxes = torch.cat([t['boxes'] for t in targets], dim=0)  # [b, 4] or [b*t,4]
        try:
            tgt_boxes_flat = tgt_boxes.view(-1, 4)
        except Exception:
            tgt_boxes_flat = tgt_boxes

        # convert to xyxy for IoU
        pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_flat)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes_flat)

        # compute pairwise GIoU between each pred and each tgt
        try:
            giou = box_ops.generalized_box_iou(pred_xyxy, tgt_xyxy)
            # giou shape: [num_preds, num_tgts]
            # compute best giou per pred
            best_giou_vals, best_tgt_idx = giou.max(dim=1)
        except Exception as e:
            print('Error computing GIoU:', e)
            giou = None

        # compute L1 to first target per pred as quick proxy
        if tgt_xyxy.numel() > 0:
            first_tgt = tgt_boxes_flat[0].unsqueeze(0).expand(pred_boxes_flat.size(0), -1)
            l1 = F.l1_loss(pred_boxes_flat, first_tgt, reduction='none').mean(1)
        else:
            l1 = None

        print('\nSample of predicted boxes (first 10 queries):')
        for i in range(min(10, pred_boxes_flat.shape[0])):
            pb = pred_boxes_flat[i].cpu().numpy()
            gi = float(best_giou_vals[i].cpu()) if giou is not None else None
            l1v = float(l1[i].cpu()) if l1 is not None else None
            print(f'  pred[{i}]: cxcywh={pb}  best_giou={gi:.4f}  l1_to_first_tgt={l1v:.6f}')

        # run full loss
        loss_dict = criterion(outputs, targets)
        print('\nComputed losses:')
        for k, v in loss_dict.items():
            print(f"  {k}: {float(v):.6f}")

        # print matcher indices from criterion.matcher for insight
        try:
            indices = criterion.matcher({k: v for k, v in outputs.items() if k != 'aux_outputs'}, targets)
            print('\nMatcher indices:')
            print(indices)
        except Exception as e:
            print('Error computing matcher indices:', e)

    except Exception as e:
        print('Error in detailed debug prints:', e)


if __name__ == '__main__':
    main()
