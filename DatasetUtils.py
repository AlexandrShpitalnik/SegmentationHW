import torch
import pydicom
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import torchvision.transforms.functional as F


class RSNADataset(Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width, train=True, transforms_list=None,
                 add_mask=False):
        self.image_fps = image_fps
        self.image_annotations = image_annotations
        self.orig_height = orig_height
        self.orig_width = orig_width
        self.transforms_list = transforms_list
        self.add_mask = add_mask

        image_info = dict()
        for image_idx, file_path in enumerate(image_fps):
            annotations = image_annotations[file_path]
            image_info[image_idx] = {"path": file_path,
                                     "annotations": annotations}
        self.image_info = image_info

    def __len__(self):
        return len(self.image_fps)

    def show_image(self, image_id):
        info = self.image_info[image_id]
        fp = info["path"]
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info["path"]
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        image = np.array(Image.fromarray(image).resize((self.orig_width, self.orig_height)))

        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        image = np.rollaxis(image, 2, 0) / 255
        return image

    def load_bbox(self, image_id, scale_factor):
        info = self.image_info[image_id]
        annotations = info["annotations"]
        count = len(annotations)
        if count == 0 or all((ann["Target"] == 0 for ann in annotations)):
            # Пневмонии нет, считаем за объект все фото
            xmin = 0
            xmax = 1024 * scale_factor
            ymin = 0
            ymax = 1024 * scale_factor
            boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            boxes = []
            mask = np.zeros((self.orig_height, self.orig_width, count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for annotation_num, annotation in enumerate(annotations):
                if annotation["Target"] == 1:
                    x = int(annotation["x"])
                    y = int(annotation["y"])
                    w = int(annotation["width"])
                    h = int(annotation["height"])
                    xmin = int(max(x * scale_factor, 0))
                    xmax = int(min((x + w) * scale_factor, self.orig_width))
                    ymin = int(max(y * scale_factor, 0))
                    ymax = int(min((y + h) * scale_factor, self.orig_height))
                    box = [xmin, ymin, xmax, ymax]

                    boxes.append(box)
                    class_ids[annotation_num] = 1
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return boxes, class_ids.astype(np.int32)

    def mask_to_bbox(self, masks, scale_factor):
        boxes = []
        for bbox in masks:
            pos = np.where(bbox[:, :])
            xmin = int(max(np.min(pos[1]) * scale_factor, 0))
            xmax = int(min(np.max(pos[1]) * scale_factor, self.orig_width))
            ymin = int(max(np.min(pos[0]) * scale_factor, 0))
            ymax = int(min(np.max(pos[0]) * scale_factor, self.orig_height))
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
            if xmin >= xmax:
                print(xmin, xmax)
            if ymin >= ymax:
                print(ymin, ymax)
            assert xmin < xmax
            assert ymin < ymax

        torch_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return torch_boxes

    @staticmethod
    def get_area(boxes):
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return area

    def get_semantic_masks(self, boxes, labels):
        if self.add_mask and np.any(labels == 1):
            inst_num = boxes.shape[0]
            masks = []
            for i in range(inst_num):
                mask_instance = np.zeros((1024, 1024), dtype=np.uint8)
                xmin, ymin, xmax, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                mask = cv2.rectangle(mask_instance, (xmin, ymin), (xmax, ymax), 255, -1)
                masks.append(mask)
            masks = np.array(masks, dtype=np.bool)
        else:
                masks = np.zeros((1024, 1024, len(labels)), dtype=np.bool)
        return torch.as_tensor(masks, dtype=torch.uint8)

    def apply_transforms(self, img, boxes, labels):
        if self.transforms_list and np.any(labels == 1):
            points = []
            for box in boxes:
                points.append([box[0], box[1]])
                points.append([box[2], box[3]])
            img, points = self.transforms_list.run(img, points)
            boxes = []
            cur_box = []
            for p in points:
                cur_box.append(p[0])
                cur_box.append(p[1])
                if len(cur_box) == 4:
                    boxes.append(cur_box)
                    cur_box = []
            return img, torch.as_tensor(boxes, dtype=torch.float32)
        elif self.transforms_list:
            img, _ = self.transforms_list.run(img, [[0, 0]])
            return img, boxes
        else:
            return img, boxes


    def __getitem__(self, index):
        scale_factor = self.orig_width / 1024

        boxes, labels = self.load_bbox(index, scale_factor)
        img = torch.Tensor(self.load_image(index))

        img, boxes = self.apply_transforms(img, boxes, labels)

        semantic_masks = self.get_semantic_masks(boxes, labels)

        if np.all(labels == 0):
            area = self.get_area(boxes)
            iscrowd = torch.ones((len(boxes),), dtype=torch.int64)
            labels = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            area = self.get_area(boxes)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {"image_id": torch.tensor([index]),
                  "boxes": boxes,
                  "labels": labels,
                  "area": area,
                  "iscrowd": iscrowd,
                  "masks": semantic_masks
                  }

        return img, target


#visualize source/transformed
def visualize_pair(dataset):
    class_ids = [0]
    while class_ids[0] == 0:
        image_id = random.choice(range(len(dataset.image_fps)))
        image_fp = dataset.image_fps[image_id]
        orig_image = dataset.show_image(image_id)
        orig_boxes, class_ids = dataset.load_bbox(image_id, scale_factor=1)
        if len(orig_boxes) == 0:
            continue
        mask_instance = np.zeros((dataset.orig_height, dataset.orig_width), dtype=np.uint8)
        xmin, ymin, xmax, ymax = orig_boxes[0][0], orig_boxes[0][1], orig_boxes[0][2], orig_boxes[0][3]
        orig_mask = cv2.rectangle(mask_instance, (xmin, ymin), (xmax, ymax), 255, -1)
        orig_mask = np.array(orig_mask)

    masked = orig_image[:, :, 0] * orig_mask
    orig_image = orig_image[:, :, 0] - masked

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image, cmap='gray')
    plt.axis('off')

    sample = dataset[image_id]
    aug_tensor = sample[0]
    aug_boxes = sample[1]['boxes']
    aug_image = np.array(F.to_pil_image(aug_tensor))
    mask_instance = np.zeros((dataset.orig_height, dataset.orig_width), dtype=np.uint8)
    xmin, ymin, xmax, ymax = aug_boxes[0][0], aug_boxes[0][1], aug_boxes[0][2], aug_boxes[0][3]
    aug_mask = cv2.rectangle(mask_instance, (xmin, ymin), (xmax, ymax), 255, -1)
    aug_mask = np.array(aug_mask)


    plt.subplot(1, 2, 2)
    masked = aug_image[:, :, 0] * aug_mask

    aug_image = aug_image[:, :, 0] - masked

    plt.imshow(aug_image, cmap='gray')
    plt.axis('off')

