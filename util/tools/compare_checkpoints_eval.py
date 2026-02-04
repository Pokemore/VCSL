import argparse
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append('/data/LQVG')

import opts
from datasets import build_dataset
from torch.utils.data import DataLoader
from models import build_model
import util.misc as utils


def tensor_stats(t):
    a = t.detach().cpu().numpy()
    return {
        'shape': list(t.shape),
        'min': float(a.min()),
        'max': float(a.max()),
        'mean': float(a.mean()),
        'std': float(a.std())
    }


def print_stats(prefix, logits, boxes):
    # logits: [t, q, k] or [1, q, k]
    print(f"--- {prefix} ---")
    print("logits stats:")
    s = tensor_stats(logits)
    print(s)
    # also print sigmoid range
    probs = logits.sigmoid()
    print("probs stats:", tensor_stats(probs))
    # boxes: [t, q, 4] or [1, q, 4]
    print("boxes stats (cx,cy,w,h):")
    print(tensor_stats(boxes))
    # centers range
    centers = boxes[..., :2]
    print("centers stats:", tensor_stats(centers))
    print("centers min/max per dim:", float(centers[...,0].min().cpu()), float(centers[...,0].max().cpu()), float(centers[...,1].min().cpu()), float(centers[...,1].max().cpu()))


def load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # prune keys in checkpoint that have mismatched shapes compared to model
    model_state = model.state_dict()
    removed = []
    if 'model' in ckpt:
        for k in list(ckpt['model'].keys()):
            if k in model_state:
                if ckpt['model'][k].shape != model_state[k].shape:
                    removed.append((k, ckpt['model'][k].shape, model_state[k].shape))
                    del ckpt['model'][k]
    else:
        # legacy: checkpoint may itself be the state_dict
        for k in list(ckpt.keys()):
            if k in model_state and hasattr(ckpt[k], 'shape'):
                if ckpt[k].shape != model_state[k].shape:
                    removed.append((k, ckpt[k].shape, model_state[k].shape))
                    del ckpt[k]

    if len(removed) > 0:
        print(f"Pruned {len(removed)} mismatched keys from checkpoint {ckpt_path}:")
        for k, s_from, s_expected in removed:
            print(f"  - {k}: checkpoint {s_from} -> model expects {s_expected}")

    missing, unexpected = model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
    print(f"Loaded {ckpt_path} -> missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt1', required=True)
    parser.add_argument('--ckpt2', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset', default='rsvg')
    args = parser.parse_args()

    # build minimal args from opts
    parser2 = opts.get_args_parser()
    parsed = parser2.parse_args([])
    parsed.dataset_file = args.dataset
    parsed.device = args.device
    parsed.batch_size = 1
    parsed.num_workers = 0
    parsed.masks = False

    device = torch.device(parsed.device)

    test_ds = build_dataset(parsed.dataset_file, image_set='test', args=parsed)
    dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # get first sample
    it = iter(dl)
    sample = next(it)
    img, targets, dw, dh, img_path, ratio = sample
    captions = targets['caption']
    h_resize, w_resize = img.shape[-2:]
    size = torch.as_tensor([int(h_resize), int(w_resize)]).to(device)
    target = {"size": size}

    # helper run for a checkpoint
    def run_ckpt(ckpt_path):
        model, criterion, _ = build_model(parsed)
        model.to(device)
        load_checkpoint_into_model(model, ckpt_path, device)
        model.eval()

        img_cuda = img.to(device)

        # run in eval
        with torch.no_grad():
            out_eval = model(img_cuda, captions, [target])

        # run in train mode (no grad) to compare BN behavior
        model.train()
        with torch.no_grad():
            out_train = model(img_cuda, captions, [target])

        # pick first elements
        logits_eval = out_eval['pred_logits'][0]
        boxes_eval = out_eval['pred_boxes'][0]
        logits_train = out_train['pred_logits'][0]
        boxes_train = out_train['pred_boxes'][0]

        return (logits_eval, boxes_eval, logits_train, boxes_train)

    print('Running ckpt1:', args.ckpt1)
    stats1 = run_ckpt(args.ckpt1)
    print_stats('CKPT1 EVAL', stats1[0], stats1[1])
    print_stats('CKPT1 TRAIN', stats1[2], stats1[3])

    print('\nRunning ckpt2:', args.ckpt2)
    stats2 = run_ckpt(args.ckpt2)
    print_stats('CKPT2 EVAL', stats2[0], stats2[1])
    print_stats('CKPT2 TRAIN', stats2[2], stats2[3])

    print('\nDone')
