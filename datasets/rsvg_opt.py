#!/usr/bin/env python
# -*-coding: utf-8-*-
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from pathlib import Path

import util
import datasets.transforms_image as T

def filelist(root, file_type):
    return [os.path.join(directory_path, f) for directory_path, directory_name, files in os.walk(root) for f in files if f.endswith(file_type)]

class RefOPTDataset(data.Dataset):
    def __init__(self, images_path, anno_path, imsize=800, transform=None, augment=False,
                 split='train', testmode=False):
        self.images = []
        self.images_path = images_path
        self.anno_path = anno_path
        self.imsize = imsize # 这里的imsize通常是 max_size
        self.augment = augment
        self.transform = transform
        self.split = split
        self.testmode = testmode

        annotations = filelist(anno_path, '.xml')
        for anno_file_path in annotations:
            try:
                root = ET.parse(anno_file_path).getroot()

                filename_node = root.find("./filename")
                if filename_node is None:
                    print(f"Warning: No 'filename' found in {anno_file_path}, skipping.")
                    continue
                image_filename = filename_node.text
                image_full_path = os.path.join(str(images_path), image_filename)

                size_node = root.find("size")
                if size_node is None:
                    print(f"Warning: No 'size' found in {anno_file_path}, skipping.")
                    continue
                img_width = int(size_node.find("width").text)
                img_height = int(size_node.find("height").text)

                for member in root.findall('object'):
                    name_node = member.find('name')
                    bndbox_node = member.find('bndbox')
                    description_node = member.find('description')

                    if name_node is None or bndbox_node is None or description_node is None:
                        print(f"Warning: Missing 'name', 'bndbox' or 'description' in an object in {anno_file_path}, skipping this object.")
                        continue

                    obj_name = name_node.text

                    xmin_orig = int(bndbox_node.find('xmin').text)
                    ymin_orig = int(bndbox_node.find('ymin').text)
                    xmax_orig = int(bndbox_node.find('xmax').text)
                    ymax_orig = int(bndbox_node.find('ymax').text)

                    # --- 改进的边界框处理逻辑 START ---

                    # 1. 将边界框坐标裁剪到图像内部
                    xmin_clamped = max(0, xmin_orig)
                    ymin_clamped = max(0, ymin_orig)
                    xmax_clamped = min(img_width, xmax_orig)
                    ymax_clamped = min(img_height, ymax_orig)

                    # 2. 检查裁剪后的边界框是否仍然有效（宽度和高度必须大于0）
                    # 添加一个最小尺寸阈值，例如1像素
                    min_bbox_dim = 1
                    if (xmax_clamped - xmin_clamped < min_bbox_dim) or \
                       (ymax_clamped - ymin_clamped < min_bbox_dim):
                        print(f"Warning: After clamping, bounding box [{xmin_orig}, {ymin_orig}, {xmax_orig}, {ymax_orig}] "
                              f"for image {image_filename} (size: {img_width}x{img_height}) becomes too small or invalid: "
                              f"[{xmin_clamped}, {ymin_clamped}, {xmax_clamped}, {ymax_clamped}], skipping this object.")
                        continue # 如果裁剪后仍然无效或过小，则跳过

                    # 如果经过裁剪后有效，则使用裁剪后的坐标
                    bbox = np.array([xmin_clamped, ymin_clamped, xmax_clamped, ymax_clamped], dtype=np.float32)

                    # 打印Info消息，显示哪些框被调整过
                    if xmin_clamped != xmin_orig or ymin_clamped != ymin_orig or \
                       xmax_clamped != xmax_orig or ymax_clamped != ymax_orig:
                        print(f"Info: Bounding box [{xmin_orig}, {ymin_orig}, {xmax_orig}, {ymax_orig}] for image {image_filename} "
                              f"(size: {img_width}x{img_height}) was clamped to [{xmin_clamped}, {ymin_clamped}, {xmax_clamped}, {ymax_clamped}].")

                    # --- 改进的边界框处理逻辑 END ---

                    text_description = description_node.text

                    # 存储 (图片完整路径, 边界框, 描述文本, 原始图片尺寸)
                    self.images.append((image_full_path, bbox, text_description, (img_width, img_height)))
            except Exception as e:
                print(f"Error processing {anno_file_path}: {e}, skipping.")
                continue

        print(f"Loaded {len(self.images)} samples for split: {split}")

    def pull_item(self, idx):
        img_path, bbox, phrase, original_size = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        return img, phrase, bbox, img_path, original_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, img_path, original_size = self.pull_item(idx)
        caption = " ".join(phrase.lower().split())
        w_orig, h_orig = original_size

        # 将bbox转换为模型期望的格式 (xmin, ymin, xmax, ymax) 并归一化
        # bbox 此时是 np.array，传给 transform
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0) # [1, 4]

        target = {}
        target["dataset_name"] = "RefOPT"
        target["boxes"] = bbox_tensor # 像素坐标，后续transform会处理归一化
        target["labels"] = torch.tensor([1])

        if caption is not None:
            target["caption"] = caption
            try:
                from models.LQVG import parse_spatial_relations
                parsed = parse_spatial_relations(caption)
                target['phrases'] = parsed.get('phrases', [])
                target['relations'] = parsed.get('relations', [])
            except (ModuleNotFoundError, ImportError):
                target['phrases'] = [caption]
                target['relations'] = []
            except Exception as e:
                print(f"Warning: Error parsing spatial relations for caption '{caption}': {e}")
                target['phrases'] = [caption]
                target['relations'] = []

        target["valid"] = torch.tensor([1])
        target["orig_size"] = torch.as_tensor([int(h_orig), int(w_orig)])
        target["size"] = torch.as_tensor([int(h_orig), int(w_orig)]) # 在transform之前，这里也是原始尺寸

        if self.transform is not None:
            img, target = self.transform(img, target)

        # 在 testmode 下，如果 transform 内部处理了图像和 bbox 缩放，
        # 那么 dw, dh, ratio 应该来自 transform 的返回值或者不需要外部获取。
        # 鉴于您的模型输入是 img.unsqueeze(0) 和 target，我们保持一致。
        # 如果模型在推理时确实需要原始图像路径或缩放参数，可能需要调整 transform 的设计。
        if self.testmode:
            # 暂时保持返回 None，因为 make_refopt_transforms 不会返回这些额外参数
            return img.unsqueeze(0), target, None, None, img_path, None
        else:
            return img.unsqueeze(0), target


def make_refopt_transforms(image_set, cautious=False):
    """
    根据图像集类型 (train/val/test) 创建相应的图像变换。
    重点是 RandomResize 会自动处理图像和边界框的缩放。
    """
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 训练时可以有多种缩放尺度进行数据增强
    scales = [480, 560, 640, 720, 800] # 最短边缩放到这些值
    max_size = 1333 # 最长边限制，这是一个常用的值，允许图片更大一些，以适应大尺寸图像

    if image_set == "train":
        return T.Compose(
            [T.RandomResize(scales, max_size=max_size), # T.RandomResize 会自动调整 target['boxes']
             normalize]
        )
    elif image_set == "val" or image_set == "test":
        # 验证和测试通常使用固定的缩放策略，比如只将最短边缩放到某个值，同时限制最长边
        # T.RandomResize([size], max_size=max_size) 可以实现这种行为
        # 这里为了适应大尺寸图像，也可以将 max_size 调大一点，比如 1333
        return T.Compose([
            T.RandomResize([800], max_size=max_size), # 例如，将最短边缩放到800，最长边不超过1333
            normalize
        ])
    else:
        raise ValueError(f"unknown {image_set}")

def build_refopt_dataset(image_set, args):
    root = Path(args.refopt_path)
    assert root.exists(), f'Provided RefOPT path {root} does not exist'

    img_folder = root / "images"
    ann_folder = root / "annotations"

    dataset = RefOPTDataset(
        images_path=img_folder,
        anno_path=ann_folder,
        transform=make_refopt_transforms(image_set),
        split=image_set,
        testmode=(image_set == 'test')
    )
    return dataset