"""생성/판별 모델 로더 + 경량 품질 지표.

무거운 모델은 스크립트가 필요할 때만 로드하고 끝나면 프로세스 종료로 VRAM을
반환한다 (8GB 랩탑 GPU에서 SDXL-Turbo + CLIP을 동시에 상주시키지 않기 위함).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

SDXL_TURBO_ID = "stabilityai/sdxl-turbo"
CLIP_ID = "openai/clip-vit-base-patch32"


def load_sdxl_turbo(device: str = "cuda"):
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_TURBO_ID, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


class ClipScorer:
    """이미지-텍스트 코사인 유사도를 softmax로 정규화해 확률처럼 쓴다."""

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def classify(self, image: Image.Image, labels: list[str]) -> dict[str, float]:
        inputs = self.processor(text=labels, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits_per_image  # (1, n_labels)
        probs = logits.softmax(dim=-1)[0].cpu().numpy()
        return dict(zip(labels, (float(p) for p in probs)))


def average_hash(image: Image.Image, hash_size: int = 8) -> int:
    """단순 aHash. imagehash 의존성 없이 근접 중복 판정용."""
    small = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)
    bits = arr > arr.mean()
    out = 0
    for b in bits.flatten():
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
