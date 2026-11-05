"""CCTV 화면처럼 보이게 만드는 생성형 스타일화 + 감시카메라 화면 표시(OSD).

8차(사용자 요청: "CCTV 화면으로 보이게 세팅하는 생성형 모델 도입").

1. 생성형 톤 변환  SDXL-Turbo img2img 로 프레임을 "감시카메라 녹화 영상"으로 다시 그린다.
   다만 프레임 전체를 그대로 다시 그리면 앞 단계에서 신원 조건부로 그려 넣은 얼굴의
   신원이 바뀐다. 그래서 생성은 저해상도(768 폭)에서 하고, 그 결과의 **저주파 성분
   (색감·조명·화이트밸런스·노출·비네팅)만** 원해상도 프레임으로 옮긴다:
       out = frame + up(blur(gen_lr) - blur(down(frame)))
   고주파(얼굴 이목구비)는 원본 프레임에서 오므로 신원이 보존되고, 톤은 생성 모델이 정한다.
   이후 scripts 쪽에서 신원을 다시 검증한다.
2. 광학  약한 배럴 왜곡(광각 렌즈) + 주변 감광.
3. OSD  날짜·시각, 카메라 번호, REC 표시 — 실제 CCTV 녹화 화면의 표준 구성.
"""

from __future__ import annotations

import datetime as dt
import random

import cv2
import numpy as np
import torch
from PIL import Image

STYLE_PROMPTS = {
    False: "cctv security camera footage, surveillance camera recording, flat fluorescent lighting, "
           "slightly washed out colors, low dynamic range, realistic",
    True: "night time cctv security camera footage, surveillance camera recording, sodium street "
          "lighting, low light, sensor noise, washed out colors, realistic",
}
GEN_W = 768
LOWFREQ_SIGMA = 6.0     # 생성 해상도 기준 px — 이보다 큰 구조(조명·색 영역)만 옮긴다


class CCTVStylizer:
    def __init__(self, embeds: dict, device: str = "cuda"):
        """embeds: {prompt: (prompt_embeds, pooled)} — genmodels.load_sdxl_turbo_text_encoders 로 미리 계산."""
        from diffusers import AutoPipelineForImage2Image

        from . import genmodels

        t2i = genmodels.load_sdxl_turbo_unet_vae(device)
        self.i2i = AutoPipelineForImage2Image.from_pipe(t2i)
        self.i2i.set_progress_bar_config(disable=True)
        self.embeds = {k: (a.to(device), b.to(device)) for k, (a, b) in embeds.items()}
        self.device = device

    @torch.no_grad()
    def tone(self, bgr: np.ndarray, night: bool, strength: float, seed: int) -> np.ndarray:
        H, W = bgr.shape[:2]
        gh = int(round(GEN_W * H / W / 8)) * 8
        lr = cv2.resize(bgr, (GEN_W, gh), interpolation=cv2.INTER_AREA)
        pe, ppe = self.embeds[STYLE_PROMPTS[night]]
        steps = max(2, int(np.ceil(2 / strength)))       # 실제 디노이즈 2스텝
        gen = self.i2i(image=Image.fromarray(lr[:, :, ::-1]), prompt_embeds=pe, pooled_prompt_embeds=ppe,
                       strength=strength, num_inference_steps=steps, guidance_scale=0.0,
                       generator=torch.Generator(self.device).manual_seed(seed)).images[0]
        gen = np.array(gen)[:, :, ::-1].astype(np.float32)
        delta = cv2.GaussianBlur(gen, (0, 0), LOWFREQ_SIGMA) - cv2.GaussianBlur(lr.astype(np.float32), (0, 0), LOWFREQ_SIGMA)
        delta = cv2.resize(delta, (W, H), interpolation=cv2.INTER_CUBIC)
        return np.clip(bgr.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def lens(bgr: np.ndarray, k1: float, vignette: float) -> tuple[np.ndarray, callable]:
    """배럴 왜곡 + 주변 감광. 반환: (이미지, 원좌표→왜곡좌표 변환 함수) — 라벨 박스 이동용."""
    H, W = bgr.shape[:2]
    cx, cy, r0 = W / 2, H / 2, np.hypot(W / 2, H / 2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = (xx - cx) / r0, (yy - cy) / r0
    r2 = xn * xn + yn * yn
    # 출력 픽셀(xd) ← 입력 픽셀(xu): xu = xd * (1 + k1 r_d^2) / (1 + k1)  (배럴: 중심이 부풀고 가장자리가 눌림)
    # (1 + k1) 로 나눠 모서리(r=1)가 입력 모서리에 그대로 대응하게 한다 — 처음엔 정규화 없이
    # 가장자리가 입력 바깥을 참조해 BORDER_REFLECT 로 채웠고, 프레임 오른쪽 끝에 사람이 거울처럼
    # 비친 형상이 생겼다(시연 그림에서 발견). 정규화하면 f <= 1 이라 항상 입력 안쪽만 읽는다.
    f = (1 + k1 * r2) / (1 + k1)
    mapx, mapy = (xn * f * r0 + cx).astype(np.float32), (yn * f * r0 + cy).astype(np.float32)
    out = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    vig = (1 - vignette * r2)[..., None]
    out = np.clip(out.astype(np.float32) * vig, 0, 255).astype(np.uint8)

    def fwd(x, y):
        """입력 좌표 → 출력 좌표 (고정점 반복으로 역변환)."""
        xu, yu = (x - cx) / r0, (y - cy) / r0
        xd, yd = xu, yu
        for _ in range(12):
            fd = (1 + k1 * (xd * xd + yd * yd)) / (1 + k1)
            xd, yd = xu / fd, yu / fd
        return xd * r0 + cx, yd * r0 + cy
    return out, fwd


def osd(bgr: np.ndarray, rng: random.Random, cam: int, night: bool = False) -> np.ndarray:
    out = bgr.copy()
    H, W = out.shape[:2]
    s = W / 1536
    hour = rng.choice([20, 21, 22, 23, 0, 1, 2, 3, 4]) if night else rng.randint(7, 18)
    t = dt.datetime(2026, rng.randint(1, 12), rng.randint(1, 28), hour, rng.randint(0, 59), rng.randint(0, 59))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, org in ((t.strftime("%Y-%m-%d %H:%M:%S"), (int(24 * s), int(44 * s))),
                      (f"CAM {cam:02d}", (int(W - 170 * s), int(H - 28 * s)))):
        cv2.putText(out, text, org, font, 0.9 * s, (0, 0, 0), max(1, int(4 * s)), cv2.LINE_AA)
        cv2.putText(out, text, org, font, 0.9 * s, (235, 235, 235), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.circle(out, (int(W - 120 * s), int(36 * s)), int(9 * s), (0, 0, 230), -1, cv2.LINE_AA)
    cv2.putText(out, "REC", (int(W - 102 * s), int(46 * s)), font, 0.8 * s, (235, 235, 235), max(1, int(2 * s)), cv2.LINE_AA)
    return out
