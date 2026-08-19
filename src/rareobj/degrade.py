"""저화질 감시/야생동물 활동 카메라 열화 시뮬레이션.

실제 저화질 영상이 만들어지는 물리적 순서를 그대로 따른다.

  1. 모션 블러     피사체·카메라 이동, 낮은 셔터 속도
  2. 다운스케일 왕복  센서 실효 해상도가 낮음 (다운 후 원 크기로 업샘플)
  3. 센서 노이즈    저조도 CMOS 게인 노이즈 (Gaussian + Poisson 근사)
  4. JPEG 재압축   저비트레이트 스트리밍/저장 코덱 아티팩트

각 단계는 강도 파라미터를 받고, `apply_pipeline` 은 이미지 한 장에 대해
난수로 뽑은 강도로 4단계를 순서대로 적용한다. 순서를 바꾸면 안 된다 — 예를 들어
JPEG 압축을 블러보다 먼저 하면 블록 아티팩트가 블러에 뭉개져 실제 저화질 영상과
결이 달라진다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DegradeParams:
    """한 장에 실제로 적용된 파라미터. 재현·문서화를 위해 그대로 저장한다."""

    motion_blur_len: int
    motion_blur_angle: float
    downscale_factor: float
    noise_sigma: float
    poisson_scale: float
    jpeg_quality: int

    def to_dict(self) -> dict:
        return {
            "motion_blur_len": self.motion_blur_len,
            "motion_blur_angle_deg": round(self.motion_blur_angle, 1),
            "downscale_factor": round(self.downscale_factor, 3),
            "noise_sigma": round(self.noise_sigma, 2),
            "poisson_scale": round(self.poisson_scale, 1),
            "jpeg_quality": self.jpeg_quality,
        }


# 실외 감시·야생동물 트레일캠 저화질 영상 관측 범위 근사. 5/21 조사한
# 디블러링/업스케일링 선행연구가 다루는 열화 강도 구간과 겹치도록 잡았다.
RANGES = {
    "motion_blur_len": (3, 15),          # px
    "downscale_factor": (0.18, 0.45),    # 원 해상도 대비
    "noise_sigma": (2.0, 10.0),          # 8bit 스케일 표준편차
    "poisson_scale": (40.0, 180.0),      # 작을수록 shot noise 강함
    "jpeg_quality": (12, 45),
}


def sample_params(rng: random.Random) -> DegradeParams:
    return DegradeParams(
        motion_blur_len=rng.randint(*RANGES["motion_blur_len"]),
        motion_blur_angle=rng.uniform(0, 180),
        downscale_factor=rng.uniform(*RANGES["downscale_factor"]),
        noise_sigma=rng.uniform(*RANGES["noise_sigma"]),
        poisson_scale=rng.uniform(*RANGES["poisson_scale"]),
        jpeg_quality=rng.randint(*RANGES["jpeg_quality"]),
    )


def _motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    if length <= 1:
        return img
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    k /= max(k.sum(), 1e-6)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT101)


def _downscale_roundtrip(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def sensor_noise(img: np.ndarray, sigma: float, poisson_scale: float) -> np.ndarray:
    """저조도 CMOS 노이즈 근사: 신호 의존 Poisson(shot) + 상수 Gaussian(read noise)."""
    f = img.astype(np.float32)
    lam = np.clip(f, 1e-3, 255) * (poisson_scale / 255.0)
    shot = np.random.poisson(lam).astype(np.float32) * (255.0 / poisson_scale)
    gauss = np.random.normal(0, sigma, f.shape).astype(np.float32)
    return np.clip(shot + gauss, 0, 255).astype(np.uint8)


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def apply_pipeline(img_bgr: np.ndarray, params: DegradeParams) -> np.ndarray:
    """BGR uint8 이미지에 4단계 열화를 순서대로 적용한다."""
    out = _motion_blur(img_bgr, params.motion_blur_len, params.motion_blur_angle)
    out = _downscale_roundtrip(out, params.downscale_factor)
    out = sensor_noise(out, params.noise_sigma, params.poisson_scale)
    out = _jpeg_roundtrip(out, params.jpeg_quality)
    return out


def degrade_image(img_bgr: np.ndarray, rng: random.Random) -> tuple[np.ndarray, DegradeParams]:
    params = sample_params(rng)
    return apply_pipeline(img_bgr, params), params
