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


def load_sdxl_turbo(device: str = "cuda", offload: bool = False):
    # 환경에 torch(2.7.0+cu128)와 ABI가 맞지 않는 xformers 0.0.32 가 깔려 있으면
    # diffusers 가 attention_processor 에서 xformers.ops 를 import 하다 DLL 로드
    # 실패로 죽는다(8차에서 겪음). xformers 는 쓰지 않으므로(SDPA로 충분) 패키지를
    # 건드리지 않고 diffusers 의 감지 플래그만 끈다 — 파이프라인 import 전에 해야 한다.
    import diffusers.utils.import_utils as _iu

    _iu._xformers_available = False
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_TURBO_ID, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.set_progress_bar_config(disable=True)
    if offload:
        # 전체 상주 시 7.05GB, 생성 피크 8.2GB — 8GB 랩탑 GPU에서 다른 프로그램이 VRAM을
        # 조금만 써도 공유메모리로 넘쳐 장당 60초+로 느려진다(8차에서 겪음). 텍스트
        # 인코더/UNet/VAE를 쓰는 순간에만 GPU로 올린다.
        pipe.enable_model_cpu_offload()
        return pipe
    return pipe.to(device)


def _disable_xformers():
    import diffusers.utils.import_utils as _iu

    _iu._xformers_available = False


def load_sdxl_turbo_text_encoders(device: str = "cuda"):
    """프롬프트 인코딩 전용 (UNet/VAE 없음). 8차 — 아래 unet_vae 로더와 짝.

    한 파이프라인에 전부 올리면 7.05GB + 생성 피크로 8GB를 넘는다. 게다가 생성 도중
    text_encoder 를 None 으로 떼어내도 내부 참조가 남아 1.6GB가 GPU에서 안 내려갔다.
    그래서 아예 두 개의 파이프라인으로 나눠, 이걸로 임베딩을 다 뽑고 del 한 뒤
    load_sdxl_turbo_unet_vae 를 쓴다."""
    _disable_xformers()
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_TURBO_ID, unet=None, vae=None, torch_dtype=torch.float16, variant="fp16")
    return pipe.to(device)


SDXL_VAE_FP16_FIX_ID = "madebyollin/sdxl-vae-fp16-fix"


def load_sdxl_turbo_unet_vae(device: str = "cuda"):
    """텍스트 인코더 없이 UNet + VAE만 (prompt_embeds 로 호출). 상주 5.4GB, 768x512 피크 6.4GB.

    VAE는 sdxl-vae-fp16-fix 로 교체한다 — 원본 SDXL VAE는 fp16에서 NaN이 나 diffusers가
    디코딩을 fp32로 올리는데(force_upcast), 그 순간 메모리가 튀어 8GB를 넘겼다."""
    _disable_xformers()
    from diffusers import AutoencoderKL, AutoPipelineForText2Image

    vae = AutoencoderKL.from_pretrained(SDXL_VAE_FP16_FIX_ID, torch_dtype=torch.float16)
    pipe = AutoPipelineForText2Image.from_pretrained(
        SDXL_TURBO_ID, vae=vae, text_encoder=None, text_encoder_2=None,
        torch_dtype=torch.float16, variant="fp16")
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
