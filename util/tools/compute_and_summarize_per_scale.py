#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from PIL import Image
import shutil


def iou_xyxy(a, b):
    # a,b are [x1,y1,x2,y2]
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = areaA + areaB - inter
    return inter / union if union>0 else 0.0


def cxcywh_to_xyxy(box, W, H):
    cx, cy, w, h = box
    x1 = (cx - w/2) * W
    y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W
    y2 = (cy + h/2) * H
    return [x1, y1, x2, y2]


def compute_for_summary(summary_path, out_summary_path, collect_dir):
    data = json.load(open(summary_path))
    stats = []
    entries = []
    for item in data:
        img = item['image']
        out_img = item['out_image']
        # recover image size from out_image existence
        # assume original images are in same dir as before
        # load the pil image to get size
        W,H = Image.open(img).size
        b1 = cxcywh_to_xyxy(item['ckpt1_box'], W, H)
        b2 = cxcywh_to_xyxy(item['ckpt2_box'], W, H)
        i = iou_xyxy(b1, b2)
        # center distance (normalized): euclidean distance between centers / sqrt(2)
        c1 = ((b1[0]+b1[2])/2/W, (b1[1]+b1[3])/2/H)
        c2 = ((b2[0]+b2[2])/2/W, (b2[1]+b2[3])/2/H)
        cd = np.linalg.norm(np.array(c1)-np.array(c2))
        entries.append({'image': img, 'out_image': out_img, 'iou': i, 'center_dist': float(cd)})
        stats.append(i)

    import statistics
    summary = {
        'n': len(entries),
        'iou_mean': statistics.mean(stats) if stats else 0.0,
        'iou_median': statistics.median(stats) if stats else 0.0,
        'iou_zero_count': sum(1 for x in stats if x==0.0),
    }

    Path(collect_dir).mkdir(parents=True, exist_ok=True)
    # pick top-5 improvements (largest IoU) and bottom-5 (smallest IoU)
    entries_sorted = sorted(entries, key=lambda x: x['iou'], reverse=True)
    top5 = entries_sorted[:5]
    bot5 = entries_sorted[-5:]

    # copy images
    for i, e in enumerate(top5):
        dst = Path(collect_dir) / f'top_{i}_{Path(e["image"]).stem}.png'
        shutil.copy(e['out_image'], dst)
    for i, e in enumerate(bot5):
        dst = Path(collect_dir) / f'bot_{i}_{Path(e["image"]).stem}.png'
        shutil.copy(e['out_image'], dst)

    open(out_summary_path, 'w').write(json.dumps({'summary': summary, 'top5': top5, 'bot5': bot5}, indent=2))
    print('Wrote', out_summary_path)


if __name__ == '__main__':
    import sys
    base = Path('/data/LQVG/tools/compare_outputs')
    summary_path = base / 'per_scale_disable_all' / 'summary_disable_all.json'
    out_summary_path = base / 'per_scale_summary.json'
    collect_dir = base / 'per_scale_summary'
    compute_for_summary(str(summary_path), str(out_summary_path), str(collect_dir))
