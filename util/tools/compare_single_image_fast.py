import argparse
import sys
from pathlib import Path
sys.path.append('/data/LQVG')

import torch
import numpy as np
from PIL import Image

import opts
from models import build_model
import util.misc as utils
from util import transforms as utrans
import datasets.transforms_image as T
from util.misc import nested_tensor_from_videos_list


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
    print(f"--- {prefix} ---")
    print("logits stats:")
    s = tensor_stats(logits)
    print(s)
    probs = logits.sigmoid()
    print("probs stats:", tensor_stats(probs))
    print("boxes stats (cx,cy,w,h):")
    print(tensor_stats(boxes))
    centers = boxes[..., :2]
    print("centers stats:", tensor_stats(centers))
    print("centers min/max per dim:", float(centers[...,0].min().cpu()), float(centers[...,0].max().cpu()), float(centers[...,1].min().cpu()), float(centers[...,1].max().cpu()))


def load_checkpoint_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_state = model.state_dict()
    removed = []
    if 'model' in ckpt:
        for k in list(ckpt['model'].keys()):
            if k in model_state:
                if ckpt['model'][k].shape != model_state[k].shape:
                    removed.append((k, ckpt['model'][k].shape, model_state[k].shape))
                    del ckpt['model'][k]
    else:
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
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)

    # minimal args
    parser2 = opts.get_args_parser()
    parsed = parser2.parse_args([])
    parsed.dataset_file = 'rsvg'
    parsed.device = args.device
    parsed.batch_size = 1
    parsed.num_workers = 0
    parsed.masks = False

    # build transform: replicate RSVG test transform + letterbox
    imsize = 800
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=mean, std=std)

    # load image and preprocess
    pil = Image.open(args.image).convert('RGB')
    np_img = np.array(pil)
    img_lb, _, ratio, dw, dh = utrans.letterbox(np_img, None, imsize)
    # to tensor + normalize
    img_t, _ = to_tensor(img_lb, None)
    img_n, _ = normalize(img_t, None)
    # add time dim like dataset: [T=1, C, H, W]
    img_n = img_n.unsqueeze(0)
    h_resize, w_resize = img_n.shape[-2:]
    size = torch.as_tensor([int(h_resize), int(w_resize)]).to(device)
    target = {"size": size}
    captions = ["a person with phrase"]

    def run_ckpt(ckpt_path, tag):
        model, criterion, _ = build_model(parsed)
        model.to(device)
        load_checkpoint_into_model(model, ckpt_path)
        model.eval()
        # wrap image tensor into NestedTensor like the dataloader does
        samples_eval = nested_tensor_from_videos_list([img_n.to(device)])
        samples_train = nested_tensor_from_videos_list([img_n.to(device)])
        with torch.no_grad():
            out_eval = model(samples_eval, captions, [target])
        model.train()
        with torch.no_grad():
            out_train = model(samples_train, captions, [target])
        logits_eval = out_eval['pred_logits'][0]
        boxes_eval = out_eval['pred_boxes'][0]
        logits_train = out_train['pred_logits'][0]
        boxes_train = out_train['pred_boxes'][0]
        print(f"\n=== Results for {tag} ===")
        print_stats(f"{tag} EVAL", logits_eval, boxes_eval)
        print_stats(f"{tag} TRAIN", logits_train, boxes_train)

    run_ckpt(args.ckpt1, 'CKPT1')
    run_ckpt(args.ckpt2, 'CKPT2')

    print('\nDone')
