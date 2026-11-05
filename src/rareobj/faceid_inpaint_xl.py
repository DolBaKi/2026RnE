"""v4 신원 조건부 인페인팅 — 장면을 그린 모델(RealVisXL)이 사람도 직접 그린다.

v3(faceid_inpaint.py)의 문제 (사용자 확인 + 진단, 2026-10-29):
  - 장면은 SDXL-Turbo, 사람은 SD1.5 inpainting 이 그려서 질감·초점이 달랐다. SDXL-Turbo 는
    CFG 가 없어 negative prompt 가 안 듣고 군중을 늘 보케·모션블러로 그린다 → 넣은 얼굴만
    선명해 "선명한 얼굴 = 타깃" 이라는 지름길이 생겼다.
  - 41b 의 저주파 톤 옮기기가 얼굴을 11~25 어둡게, 주변을 밝게 만들어 머리 둘레에 흰 테두리.
v4 는 장면과 인물을 같은 RealVisXL(SDXL) 로 그리고, 경계는 라플라시안 피라미드로 섞는다.

  생성기   RealVisXL V5.0 + IP-Adapter FaceID-PlusV2 SDXL (+ 전용 LoRA)
           조건 = ArcFace(buffalo_l) 신원 벡터 + CLIP-ViT-H 얼굴 특징 (v3 와 동일)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from .faceid_inpaint import ArcFace, FACEID_REPO, IMAGE_ENCODER_REPO, IMAGE_ENCODER_SUB, norm_crop

SCENE_ID = "SG161222/RealVisXL_V5.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
FACEID_WEIGHT_XL = "ip-adapter-faceid-plusv2_sdxl.bin"
FACEID_LORA_XL = "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"


def _disable_xformers():
    import diffusers.utils.import_utils as _iu

    _iu._xformers_available = False   # genmodels.load_sdxl_turbo 주석 참고


def load_scene_pipe():
    """RealVisXL text2img (CPU 오프로드, 1152x768 생성 피크 5.5GB, 25스텝 장당 ~16초)."""
    _disable_xformers()
    from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(SCENE_ID, vae=vae, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


class FaceIDInpainterXL:
    def __init__(self, lora_scale: float = 0.7):
        _disable_xformers()
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLInpaintPipeline
        from transformers import CLIPVisionModelWithProjection

        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16)
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            SCENE_ID, vae=vae, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IMAGE_ENCODER_REPO, subfolder=IMAGE_ENCODER_SUB, torch_dtype=torch.float16)
        pipe.load_ip_adapter(FACEID_REPO, subfolder=None, weight_name=FACEID_WEIGHT_XL, image_encoder_folder=None)
        pipe.load_lora_weights(FACEID_REPO, weight_name=FACEID_LORA_XL, adapter_name="faceid")
        pipe.set_adapters(["faceid"], adapter_weights=[lora_scale])
        pipe.fuse_lora(adapter_names=["faceid"], lora_scale=1.0)
        # fuse 만 하면 peft 의 LoRA 행렬이 UNet 에 남아 2.6B → 3.5B 가 되고, 8GB 를 넘겨 공유메모리로
        # 스왑해 삽입 1건에 10분이 걸렸다. 합친 뒤 LoRA 행렬은 떼어낸다.
        pipe.unload_lora_weights()
        # 그래도 FaceID 어댑터 레이어까지 3.1B(fp16 6.2GB) + 활성값이 8GB 를 넘는다 → 가중치는 fp8 로
        # 저장하고 계산할 때만 fp16 으로 올린다 (diffusers layerwise casting).
        pipe.unet.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.arcface = ArcFace()

    def identity_embeds(self, faces: list[tuple[np.ndarray, np.ndarray]]):
        """faces: [(bgr 이미지, 5점 랜드마크)] — 같은 사람의 사진 여러 장을 평균해 쓴다."""
        ids = self.arcface([norm_crop(b, lm, 112) for b, lm in faces])
        idv = ids.mean(0)
        idv /= np.linalg.norm(idv)
        clip_face = Image.fromarray(norm_crop(faces[0][0], faces[0][1], 224)[:, :, ::-1])
        return idv, clip_face

    @torch.no_grad()
    def inpaint(self, window_bgr: np.ndarray, mask: np.ndarray, idv: np.ndarray, clip_face: Image.Image,
                prompt: str, negative: str, seed: int, steps: int = 30, guidance: float = 5.0,
                ip_scale: float = 1.0, strength: float = 0.99, shortcut_scale: float = 1.5) -> np.ndarray:
        pipe = self.pipe
        side = window_bgr.shape[0]
        ref = torch.from_numpy(idv).float()[None, None]                     # (1,1,512)
        id_embeds = torch.cat([torch.zeros_like(ref), ref]).to("cuda", torch.float16)
        clip_embeds = pipe.prepare_ip_adapter_image_embeds([clip_face], None, torch.device("cuda"), 1, True)[0]
        proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        proj.clip_embeds = clip_embeds.to(dtype=torch.float16)
        proj.shortcut = True          # PlusV2
        proj.shortcut_scale = shortcut_scale
        pipe.set_ip_adapter_scale(ip_scale)
        out = pipe(prompt=prompt, negative_prompt=negative,
                   image=Image.fromarray(window_bgr[:, :, ::-1]),
                   mask_image=Image.fromarray((mask * 255).astype(np.uint8)),
                   ip_adapter_image_embeds=[id_embeds], num_inference_steps=steps,
                   guidance_scale=guidance, strength=strength, width=side, height=side,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        return np.array(out)[:, :, ::-1].copy()


    @torch.no_grad()
    def rerender(self, bgr: np.ndarray, prompt: str, negative: str, seed: int, strength: float = 0.25,
                 steps: int = 25, guidance: float = 5.0) -> np.ndarray:
        """프레임 전체를 원해상도에서 낮은 강도로 다시 생성 (마스크 전체 = img2img, FaceID 끔).

        얼굴만 확대 창에서 그려 축소하면 장면 해상도로 그려진 원래 얼굴과 주파수 특성이 달라
        판별기(scripts/52)가 정면 대조군 기준 AUC 0.90 으로 갈랐다. 삽입이 끝난 프레임을 같은 모델이
        같은 해상도로 한 번 더 그리면 넣은 얼굴·원래 얼굴이 같은 렌더링을 거친다."""
        pipe = self.pipe
        H, W = bgr.shape[:2]
        zero = torch.zeros(2, 1, 512, device="cuda", dtype=torch.float16)
        proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        proj.clip_embeds = torch.zeros(2, 1, 257, 1280, device="cuda", dtype=torch.float16)
        pipe.set_ip_adapter_scale(0.0)
        out = pipe(prompt=prompt, negative_prompt=negative, image=Image.fromarray(bgr[:, :, ::-1]),
                   mask_image=Image.new("L", (W, H), 255), ip_adapter_image_embeds=[zero],
                   num_inference_steps=steps, guidance_scale=guidance, strength=strength, width=W, height=H,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        return np.array(out)[:, :, ::-1].copy()


def pyramid_blend(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray, levels: int = 4) -> np.ndarray:
    """라플라시안 피라미드 블렌딩 — 저주파(색·조명)는 넓게, 고주파(질감)는 좁게 섞어 이음매를 없앤다.
    단일 알파 블렌딩은 마스크 경계에서 질감이 반투명하게 겹쳐 '붙인 티'가 남는다."""
    h, w = bg.shape[:2]
    levels = max(1, min(levels, int(np.log2(min(h, w))) - 2))
    a = alpha.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    gf, gb, ga = [fg.astype(np.float32)], [bg.astype(np.float32)], [np.repeat(a, 3, 2)]
    for _ in range(levels):
        gf.append(cv2.pyrDown(gf[-1]))
        gb.append(cv2.pyrDown(gb[-1]))
        ga.append(cv2.pyrDown(ga[-1]))
    out = gf[-1] * ga[-1] + gb[-1] * (1 - ga[-1])
    for i in range(levels - 1, -1, -1):
        size = (gf[i].shape[1], gf[i].shape[0])
        lf = gf[i] - cv2.pyrUp(gf[i + 1], dstsize=size)
        lb = gb[i] - cv2.pyrUp(gb[i + 1], dstsize=size)
        out = cv2.pyrUp(out, dstsize=size) + lf * ga[i] + lb * (1 - ga[i])
    return np.clip(out, 0, 255).astype(np.uint8)
