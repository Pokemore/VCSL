import torch_patch
import argparse
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.misc import AverageMeter
from models import build_model
from datasets import build_dataset
import opts
from PIL import Image, ImageDraw


# Simple evaluation script tailored for LQVG-style model outputs
# - loads model via build_model(args)
# - iterates test dataset (batch_size==1 assumed as in original)
# - selects the best query per image from model outputs
# - computes IoU and acc@thresholds and prints final metrics


def bbox_iou(box1, box2):
    # box: [x1,y1,x2,y2]
    b1_x1, b1_y1, b1_x2, b1_y2 = torch.tensor(box1[0]), torch.tensor(box1[1]), torch.tensor(box1[2]), torch.tensor(box1[3])
    b2_x1, b2_y1, b2_x2, b2_y2 = torch.tensor(box2[0]), torch.tensor(box2[1]), torch.tensor(box2[2]), torch.tensor(box2[3])
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    return (inter_area + 1e-6) / (union_area + 1e-6), inter_area, union_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=0)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def evaluate(test_loader, model, args, visualize=False, vis_dir="test_output"):
    device = args.device
    model.eval()

    batch_time = AverageMeter()
    acc5 = AverageMeter(); acc6 = AverageMeter(); acc7 = AverageMeter(); acc8 = AverageMeter(); acc9 = AverageMeter()
    meanIoU = AverageMeter()
    inter_area = AverageMeter(); union_area = AverageMeter()

    end = time.time()
    img_seen = set()

    for batch_idx, (img, targets, dw, dh, img_path, ratio) in enumerate(test_loader):
        h_resize, w_resize = img.shape[-2:]
        img = img.to(device)
        captions = targets["caption"]
        size = torch.as_tensor([int(h_resize), int(w_resize)]).to(device)
        target = {"size": size}

        with torch.no_grad():
            outputs = model(img, captions, [target])

        # select top scoring query per image (same logic as original)
        pred_logits = outputs["pred_logits"][0]
        pred_bbox = outputs["pred_boxes"][0]
        pred_score = pred_logits.sigmoid().squeeze(0)
        max_score, _ = pred_score.max(-1)
        _, max_ind = max_score.max(-1)
        pred_bbox = pred_bbox[0, max_ind]

        pred_bbox = rescale_bboxes(pred_bbox.detach(), (w_resize, h_resize)).numpy()
        target_bbox = rescale_bboxes(targets["boxes"].squeeze(), (w_resize, h_resize)).numpy()

        pred_bbox[0], pred_bbox[2] = (pred_bbox[0] - dw) / ratio, (pred_bbox[2] - dw) / ratio
        pred_bbox[1], pred_bbox[3] = (pred_bbox[1] - dh) / ratio, (pred_bbox[3] - dh) / ratio
        target_bbox[0], target_bbox[2] = (target_bbox[0] - dw) / ratio, (target_bbox[2] - dw) / ratio
        target_bbox[1], target_bbox[3] = (target_bbox[1] - dh) / ratio, (target_bbox[3] - dh) / ratio

        if visualize:
            src_img = Image.open(img_path[0]).convert('RGB')
            draw = ImageDraw.Draw(src_img)
            xmin, ymin, xmax, ymax = [float(x) for x in pred_bbox.tolist()]
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(255, 0, 0), width=2)
            vis_path = os.path.join(vis_dir, Path(img_path[0]).name)
            os.makedirs(vis_dir, exist_ok=True)
            src_img.save(vis_path)

        iou, interArea, unionArea = bbox_iou(pred_bbox, target_bbox)
        cumInterArea = float(interArea.numpy())
        cumUnionArea = float(unionArea.numpy())

        accu5 = float((iou.numpy() > 0.5).astype(float).sum())
        accu6 = float((iou.numpy() > 0.6).astype(float).sum())
        accu7 = float((iou.numpy() > 0.7).astype(float).sum())
        accu8 = float((iou.numpy() > 0.8).astype(float).sum())
        accu9 = float((iou.numpy() > 0.9).astype(float).sum())

        meanIoU.update(float(iou.numpy().mean()), img.size(0))
        inter_area.update(cumInterArea)
        union_area.update(cumUnionArea)

        acc5.update(accu5, img.size(0))
        acc6.update(accu6, img.size(0))
        acc7.update(accu7, img.size(0))
        acc8.update(accu8, img.size(0))
        acc9.update(accu9, img.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            print(f"[{batch_idx}/{len(test_loader)}] Time {batch_time.avg:.3f} acc@0.5: {acc5.avg:.4f} acc@0.6: {acc6.avg:.4f} acc@0.7: {acc7.avg:.4f} meanIoU: {meanIoU.avg:.4f}")

    final_str = f"acc@0.5: {acc5.avg:.4f} acc@0.6: {acc6.avg:.4f} acc@0.7: {acc7.avg:.4f} acc@0.8: {acc8.avg:.4f} acc@0.9: {acc9.avg:.4f} meanIoU: {meanIoU.avg:.4f} cumuIoU: {inter_area.sum / union_area.sum:.4f}"
    print(final_str)


def main(args):
    args.masks = False
    print("Evaluate script (batch_size should be 1)")

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    test_dataset = build_dataset(args.dataset_file, image_set='test', args=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=True, num_workers=4)

    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    evaluate(test_loader, model, args, visualize=args.visualize, vis_dir=args.vis_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate RSVG', parents=[opts.get_args_parser()])
    parser.add_argument('--visualize', action='store_true', help='save visualization images')
    parser.add_argument('--vis-dir', default='test_output', help='where to save visualizations')
    args = parser.parse_args()
    main(args)
