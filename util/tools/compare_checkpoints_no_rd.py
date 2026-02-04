#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.append('/data/LQVG')

import torch
import numpy as np
from PIL import Image, ImageDraw
import json

import opts
from models import build_model
from util import transforms as utrans
import datasets.transforms_image as T
from util.misc import nested_tensor_from_videos_list


def rescale_bboxes(boxes, size):
    w, h = size
    boxes_xyxy = []
    for b in boxes:
        cx, cy, bw, bh = b
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes_xyxy.append([x1, y1, x2, y2])
    return np.array(boxes_xyxy)


def load_and_preprocess(image_path, imsize=800):
    pil = Image.open(image_path).convert('RGB')
    np_img = np.array(pil)
    img_lb, _, ratio, dw, dh = utrans.letterbox(np_img, None, imsize)
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_t, _ = to_tensor(img_lb, None)
    img_n, _ = normalize(img_t, None)
    img_n = img_n.unsqueeze(0)
    return img_n, pil.size, (ratio, dw, dh)


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


def draw_boxes_on_image(pil_img, boxes_xyxy, color, width=3):
    draw = ImageDraw.Draw(pil_img)
    for box in boxes_xyxy:
        draw.rectangle(box, outline=color, width=width)
    return pil_img


def run_no_rd(ckpt1, ckpt2, image_paths, out_dir, device='cuda:0'):
    device = torch.device(device)
    parser2 = opts.get_args_parser()
    parsed = parser2.parse_args([])
    parsed.dataset_file = 'rsvg'
    parsed.device = device
    parsed.batch_size = 1
    parsed.num_workers = 0
    parsed.masks = False

    model1, _, _ = build_model(parsed)
    model2, _, _ = build_model(parsed)
    model1.to(device)
    model2.to(device)
    load_checkpoint_into_model(model1, ckpt1)
    load_checkpoint_into_model(model2, ckpt2)

    # Disable RD in model1
    try:
        model1.rd_modules = None
        print('Disabled rd_modules in model1')
    except Exception:
        print('model1 has no rd_modules attribute or disable failed')

    model1.eval(); model2.eval()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = []

    for img_path in image_paths:
        print('Processing', img_path)
        img_tensor, orig_size, _ = load_and_preprocess(img_path)
        w_orig, h_orig = orig_size
        samples1 = nested_tensor_from_videos_list([img_tensor.to(device)])
        samples2 = nested_tensor_from_videos_list([img_tensor.to(device)])
        captions = ["a phrase"]
        size = torch.as_tensor([int(img_tensor.shape[-2]), int(img_tensor.shape[-1])]).to(device)
        target = {"size": size}

        with torch.no_grad():
            out1 = model1(samples1, captions, [target])
        with torch.no_grad():
            out2 = model2(samples2, captions, [target])

        logits1 = out1['pred_logits'][0].detach().cpu(); boxes1 = out1['pred_boxes'][0].detach().cpu()
        logits2 = out2['pred_logits'][0].detach().cpu(); boxes2 = out2['pred_boxes'][0].detach().cpu()

        scores1 = logits1.sigmoid().squeeze(0).max(-1)[0]; best_idx1 = int(scores1.argmax().item())
        scores2 = logits2.sigmoid().squeeze(0).max(-1)[0]; best_idx2 = int(scores2.argmax().item())
        box1 = boxes1[0, best_idx1].numpy() if boxes1.ndim==3 else boxes1[best_idx1].numpy()
        box2 = boxes2[0, best_idx2].numpy() if boxes2.ndim==3 else boxes2[best_idx2].numpy()

        bboxes1_xyxy = rescale_bboxes([box1], (w_orig, h_orig))
        bboxes2_xyxy = rescale_bboxes([box2], (w_orig, h_orig))

        pil = Image.open(img_path).convert('RGB'); vis = pil.copy()
        vis = draw_boxes_on_image(vis, bboxes1_xyxy.tolist(), color='red', width=4)
        vis = draw_boxes_on_image(vis, bboxes2_xyxy.tolist(), color='blue', width=2)
        out_img_path = out_dir / (Path(img_path).stem + '_compare_no_rd.png')
        vis.save(out_img_path)

        stats.append({'image': str(img_path), 'out_image': str(out_img_path), 'ckpt1_box': box1.tolist(), 'ckpt2_box': box2.tolist(), 'ckpt1_logits_mean': float(logits1.mean().item()), 'ckpt2_logits_mean': float(logits2.mean().item())})

    with open(out_dir / 'summary_no_rd.json','w') as f:
        json.dump(stats, f, indent=2)
    print('Done. Outputs saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt1', required=True)
    parser.add_argument('--ckpt2', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--out-dir', default='/data/LQVG/tools/compare_outputs/no_rd')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    imgs = sorted([str(p) for p in img_dir.glob('*.jpg')])[:20]
    run_no_rd(args.ckpt1, args.ckpt2, imgs, args.out_dir, device=args.device)
