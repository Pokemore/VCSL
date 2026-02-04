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


def load_checkpoint_meta(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    meta = {}
    if isinstance(ckpt, dict):
        # common keys
        meta['has_model'] = 'model' in ckpt
        if 'args' in ckpt:
            meta['args'] = ckpt['args']
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            keys = list(ckpt['model'].keys())
            meta['model_keys_sample'] = keys[:20]
            # capture shapes for query_embed and class_embed if present
            for k in ['query_embed.weight', 'class_embed.weight', 'class_embed.bias']:
                if k in ckpt['model']:
                    meta[k] = tuple(ckpt['model'][k].shape)
    return meta


def inspect_rd_params(model):
    info = {}
    if getattr(model, 'rd_modules', None) is None:
        info['rd_present'] = False
        return info
    info['rd_present'] = True
    info['num_rd_modules'] = len(model.rd_modules)
    per = []
    for i, rd in enumerate(model.rd_modules):
        d = {}
        # look for alpha, and any conv weight norms
        d['alpha'] = getattr(rd, 'alpha', None)
        try:
            d['D_conv_norm'] = float(rd.D.conv.weight.norm().item())
        except Exception:
            d['D_conv_norm'] = None
        # try to find offset-like params by name
        offsets = {k: tuple(v.shape) for k, v in rd.named_parameters() if 'offset' in k or 'bias' in k}
        d['offset_param_shapes'] = offsets
        per.append(d)
    info['per_rd'] = per
    return info


def draw_boxes_on_image(pil_img, boxes_xyxy, color, width=3):
    draw = ImageDraw.Draw(pil_img)
    for box in boxes_xyxy:
        draw.rectangle(box, outline=color, width=width)
    return pil_img


def run_inspect_and_ablate(ckpt1, ckpt2, image_paths, out_dir, device='cuda:0'):
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

    ckpt1_meta = load_checkpoint_meta(ckpt1)
    ckpt2_meta = load_checkpoint_meta(ckpt2)

    # load weights (with pruning as before)
    def load_and_prune(model, path):
        ckpt = torch.load(path, map_location='cpu')
        model_state = model.state_dict()
        removed = []
        data = ckpt['model'] if 'model' in ckpt else ckpt
        for k in list(data.keys()):
            if k in model_state and hasattr(data[k], 'shape'):
                if data[k].shape != model_state[k].shape:
                    removed.append((k, data[k].shape, model_state[k].shape))
                    del data[k]
        if len(removed) > 0:
            print(f"Pruned {len(removed)} mismatched keys from checkpoint {path}:")
            for k, s_from, s_expected in removed:
                print(f"  - {k}: checkpoint {s_from} -> model expects {s_expected}")
        missing, unexpected = model.load_state_dict(data, strict=False)
        return removed, missing, unexpected

    removed1, missing1, unexpected1 = load_and_prune(model1, ckpt1)
    removed2, missing2, unexpected2 = load_and_prune(model2, ckpt2)

    rd_info1 = inspect_rd_params(model1)
    rd_info2 = inspect_rd_params(model2)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'ckpt_config_check.txt','w') as f:
        f.write('CKPT1: ' + ckpt1 + '\n')
        f.write(json.dumps(ckpt1_meta, indent=2, default=str) + '\n\n')
        f.write('CKPT2: ' + ckpt2 + '\n')
        f.write(json.dumps(ckpt2_meta, indent=2, default=str) + '\n\n')
        f.write('RD INFO MODEL1:\n' + json.dumps(rd_info1, indent=2, default=str) + '\n\n')
        f.write('RD INFO MODEL2:\n' + json.dumps(rd_info2, indent=2, default=str) + '\n\n')

    # Per-scale ablation: for each rd module index, disable only that index and run compare on images
    per_scale_stats = []
    num_rd = len(model1.rd_modules) if getattr(model1, 'rd_modules', None) is not None else 0
    scales_to_test = list(range(num_rd)) if num_rd>0 else []
    # also add 'all' (disabled all) for reference
    scales_to_test.append('all')

    for s in scales_to_test:
        print('Running ablation disable scale:', s)
        # reload model weights fresh each time to avoid in-place mutation
        model1_tmp, _, _ = build_model(parsed)
        model1_tmp.to(device)
        load_and_prune(model1_tmp, ckpt1)
        if s == 'all':
            model1_tmp.rd_modules = None
        else:
            # disable only rd_modules[s]
            try:
                for i in range(len(model1_tmp.rd_modules)):
                    if i == s:
                        model1_tmp.rd_modules[i] = None
            except Exception:
                model1_tmp.rd_modules = None

        model1_tmp.eval(); model2.eval()

        stats = []
        test_out_dir = out_dir / f'per_scale_disable_{s}'
        test_out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in image_paths:
            img_tensor, orig_size, _ = load_and_preprocess(img_path)
            w_orig, h_orig = orig_size
            samples1 = nested_tensor_from_videos_list([img_tensor.to(device)])
            samples2 = nested_tensor_from_videos_list([img_tensor.to(device)])
            captions = ["a phrase"]
            size = torch.as_tensor([int(img_tensor.shape[-2]), int(img_tensor.shape[-1])]).to(device)
            target = {"size": size}

            with torch.no_grad():
                out1 = model1_tmp(samples1, captions, [target])
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
            out_img_path = test_out_dir / (Path(img_path).stem + f'_compare_disable_{s}.png')
            vis.save(out_img_path)

            stats.append({'image': str(img_path), 'out_image': str(out_img_path), 'ckpt1_box': box1.tolist(), 'ckpt2_box': box2.tolist(), 'ckpt1_logits_mean': float(logits1.mean().item()), 'ckpt2_logits_mean': float(logits2.mean().item())})

        with open(test_out_dir / f'summary_disable_{s}.json','w') as f:
            json.dump(stats, f, indent=2)
        per_scale_stats.append({'scale': s, 'summary': str(test_out_dir / f'summary_disable_{s}.json')})

    with open(out_dir / 'per_scale_ablation.json','w') as f:
        json.dump({'ckpt1': ckpt1, 'ckpt2': ckpt2, 'per_scale': per_scale_stats}, f, indent=2)

    print('Done. Outputs saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt1', required=True)
    parser.add_argument('--ckpt2', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--out-dir', default='/data/LQVG/tools/compare_outputs')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    imgs = sorted([str(p) for p in img_dir.glob('*.jpg')])[:20]
    run_inspect_and_ablate(args.ckpt1, args.ckpt2, imgs, args.out_dir, device=args.device)
