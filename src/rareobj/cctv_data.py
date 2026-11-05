"""9차 — CCTV 얼굴 검출 PyTorch Dataset (scripts/61 의 YOLO 형식 폴더를 읽는다).

한 샘플 = 프레임 1장(원본 또는 스왑본) + 얼굴 박스들. 기본 증강은 Albumentations 로 박스와 함께
변환한다. 얼굴이 중앙값 19px 이라 박스를 잃기 쉬운 증강(큰 회전·강한 크롭)은 약하게만 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_SIZE = 1024          # 973x489 프레임 긴 변 기준. 640 으로 줄이면 얼굴이 12px 로 작아진다


def basic_aug(train: bool) -> A.Compose:
    # 973x489 → 1024x515, 높이를 stride 32 배수(544)로 채운다
    resize = [A.LongestMaxSize(IMG_SIZE), A.PadIfNeeded(544, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT,
                                                         value=(114, 114, 114), position="top_left")]
    if not train:
        tf = resize
    else:
        tf = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.1, 0.25), rotate_limit=3,
                               border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.5),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.6),
            A.HueSaturationValue(5, 20, 15, p=0.4),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 7))], p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.Downscale(scale_min=0.5, scale_max=0.85, interpolation=cv2.INTER_LINEAR, p=0.25),
            A.ImageCompression(quality_lower=40, quality_upper=90, p=0.5),
        ] + resize
    return A.Compose(tf, bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], min_visibility=0.4,
                                                  min_width=2, min_height=2))


class CCTVFaceDataset(Dataset):
    """split ∈ train/val/test. use_swap=False 면 스왑본(*_sw)을 빼서 '기본 증강만' 비교군을 만든다."""

    def __init__(self, root: Path, split: str, use_swap: bool = True, train_aug: bool | None = None):
        self.root = Path(root)
        meta = [json.loads(l) for l in open(self.root / "meta.jsonl", encoding="utf-8")]
        self.items = [m for m in meta if m["split"] == split and (use_swap or not m["swaps"])]
        self.aug = basic_aug(split == "train" if train_aug is None else train_aug)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        m = self.items[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / m["image"])), cv2.COLOR_BGR2RGB)
        lp = self.root / m["image"].replace("images/", "labels/").replace(".jpg", ".txt")
        rows = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
        boxes = np.clip(np.float32([r[1:] for r in rows]).reshape(-1, 4), 1e-6, 1.0)
        # yolo 박스가 경계에 걸치면 Albumentations 가 거부하므로 cx±w/2 를 [0,1] 안으로 줄인다
        x0, y0 = np.clip(boxes[:, 0] - boxes[:, 2] / 2, 0, 1), np.clip(boxes[:, 1] - boxes[:, 3] / 2, 0, 1)
        x1, y1 = np.clip(boxes[:, 0] + boxes[:, 2] / 2, 0, 1), np.clip(boxes[:, 1] + boxes[:, 3] / 2, 0, 1)
        boxes = np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], 1)
        out = self.aug(image=img, bboxes=boxes, cls=[0] * len(boxes))
        im = torch.from_numpy(out["image"]).permute(2, 0, 1).float().div(255)
        H, W = out["image"].shape[:2]
        b = np.float32(out["bboxes"]).reshape(-1, 4)
        xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H,
                         (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
        return im, {"boxes": torch.from_numpy(xyxy), "swapped": bool(m["swaps"]), "image": m["image"],
                    "cam": m["cam"]}


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]
