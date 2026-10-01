"""신원 조건부 인페인팅 — 특정 인물을 장면 속 사람 자리에 "AI가 직접 그려 넣는다".

8차에서 처음엔 실제 얼굴 사진을 생성 인물 위에 붙이고(정렬·색보정·img2img 경계
섞기) 썼는데, 사용자 확인 결과 본질이 "얼굴 붙이기"라 부자연스러웠다. 이 모듈은
붙이기를 하지 않는다 — 장면에서 그 인물의 머리 영역을 마스크로 비우고, 확산모델이
찾을 사람의 신원 벡터를 조건으로 그 자리의 머리·얼굴·머리카락·목 경계를 장면의
조명·각도·흐림에 맞춰 한 번에 생성한다.

  생성기   SD1.5 inpainting + IP-Adapter FaceID-PlusV2 (+ 전용 LoRA)
           조건 = ArcFace(insightface buffalo_l w600k_r50) 신원 벡터 + CLIP-ViT-H 얼굴 특징
  신원 검증 생성기와 무관한 CASIA-WebFace InceptionResnetV1 로 생성 얼굴 vs 원본 사진
           코사인을 잰다 (생성 조건에 쓴 ArcFace로 재면 자기 채점이 되므로 분리)

참고: Ye et al., IP-Adapter (arXiv:2308.06721), h94/IP-Adapter-FaceID 모델 카드.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

INPAINT_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
FACEID_REPO = "h94/IP-Adapter-FaceID"
FACEID_WEIGHT = "ip-adapter-faceid-plusv2_sd15.bin"
FACEID_LORA = "ip-adapter-faceid-plusv2_sd15_lora.safetensors"
IMAGE_ENCODER_REPO, IMAGE_ENCODER_SUB = "h94/IP-Adapter", "models/image_encoder"
ARCFACE_REPO, ARCFACE_FILE = "public-data/insightface", "models/buffalo_l/w600k_r50.onnx"

# insightface 표준 112x112 정렬 템플릿 (leye, reye, nose, lmouth, rmouth)
ARCFACE_DST = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                          [41.5493, 92.3655], [70.7299, 92.2041]])


def norm_crop(bgr: np.ndarray, lm5: np.ndarray, size: int = 112) -> np.ndarray:
    M, _ = cv2.estimateAffinePartial2D(np.float32(lm5), ARCFACE_DST * (size / 112.0), method=cv2.LMEDS)
    return cv2.warpAffine(bgr, M, (size, size), borderValue=0)


class ArcFace:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        self.sess = ort.InferenceSession(hf_hub_download(ARCFACE_REPO, ARCFACE_FILE),
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, crops112_bgr: list[np.ndarray]) -> np.ndarray:
        x = np.stack([c[:, :, ::-1] for c in crops112_bgr]).astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = self.sess.run(None, {self.inp: x})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)


class FaceIDInpainter:
    def __init__(self, device: str = "cuda", lora_scale: float = 0.7):
        import diffusers.utils.import_utils as _iu

        _iu._xformers_available = False   # genmodels.load_sdxl_turbo 주석 참고
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
        from transformers import CLIPVisionModelWithProjection

        self.device = device
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINT_ID, torch_dtype=torch.float16, variant="fp16", safety_checker=None,
            requires_safety_checker=False)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
        pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IMAGE_ENCODER_REPO, subfolder=IMAGE_ENCODER_SUB, torch_dtype=torch.float16)
        pipe.load_ip_adapter(FACEID_REPO, subfolder=None, weight_name=FACEID_WEIGHT, image_encoder_folder=None)
        pipe.load_lora_weights(FACEID_REPO, weight_name=FACEID_LORA, adapter_name="faceid")
        pipe.set_adapters(["faceid"], adapter_weights=[lora_scale])
        pipe.fuse_lora(adapter_names=["faceid"], lora_scale=1.0)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(device)
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
                prompt: str, negative: str, seed: int, steps: int = 20, guidance: float = 6.0,
                ip_scale: float = 1.0, strength: float = 1.0, shortcut_scale: float = 1.0) -> np.ndarray:
        pipe = self.pipe
        side = window_bgr.shape[0]
        ref = torch.from_numpy(idv).float()[None, None]                     # (1,1,512)
        id_embeds = torch.cat([torch.zeros_like(ref), ref]).to(self.device, torch.float16)
        clip_embeds = pipe.prepare_ip_adapter_image_embeds(
            [clip_face], None, torch.device(self.device), 1, True)[0]
        proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        proj.clip_embeds = clip_embeds.to(dtype=torch.float16)
        proj.shortcut = True          # PlusV2
        proj.shortcut_scale = shortcut_scale   # 모델 카드의 s_scale — 클수록 CLIP 얼굴 특징(신원) 강화
        pipe.set_ip_adapter_scale(ip_scale)
        out = pipe(prompt=prompt, negative_prompt=negative,
                   image=Image.fromarray(window_bgr[:, :, ::-1]),
                   mask_image=Image.fromarray((mask * 255).astype(np.uint8)),
                   ip_adapter_image_embeds=[id_embeds], num_inference_steps=steps,
                   guidance_scale=guidance, strength=strength, width=side, height=side,
                   generator=torch.Generator(self.device).manual_seed(seed)).images[0]
        return np.array(out)[:, :, ::-1].copy()
