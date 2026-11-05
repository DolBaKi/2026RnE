"""실제 CCTV 군중 프레임용 얼굴 검출(SCRFD-10G) + 얼굴 스왑(inswapper_128) — 둘 다 PyTorch(GPU).

2026-10-30 방향 전환: 장면을 생성(v4)하지 않고 실제 CCTV 프레임(data/dataset/CCTV)을 쓴다.
  SCRFD-10G      InsightFace buffalo_l det_10g. 박스 라벨이 없어 의사 라벨을 만든다.
                 얼굴 중앙값 19px 이라 프레임을 2배 키워 검출한다.
  inswapper_128  박스 안 얼굴의 신원만 바꾼다(자세·표정·조명은 원래 얼굴). 신원 = ArcFace 512 벡터.
두 ONNX 는 onnxruntime 이 CPU 판뿐이라 onnx2torch 로 변환해 GPU 에서 돌린다.
inswapper 가중치는 deepinsight 저장소가 막혀 있어 미러를 받고 공식 SHA-256 으로 확인한다.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
import torch

from .faceid_inpaint import ARCFACE_DST, ArcFace

DET_REPO, DET_FILE = "public-data/insightface", "models/buffalo_l/det_10g.onnx"
SWAP_REPO, SWAP_FILE = "ezioruan/inswapper_128.onnx", "inswapper_128.onnx"
SWAP_SHA256 = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
# inswapper 정렬: insightface estimate_norm 은 128 이 112 의 배수가 아니라 ratio=1, x 로 +8 이동
SWAP_DST = ARCFACE_DST + np.float32([8.0, 0.0])
# 128 정렬 공간의 볼·코 피부 영역 (입 제외) — 피부톤 측정·맞춤 기준
CHEEK = np.zeros((128, 128), np.uint8)
cv2.ellipse(CHEEK, (64, 74), (30, 20), 0, 0, 360, 1, -1)
cv2.rectangle(CHEEK, (44, 84), (84, 100), 0, -1)


def _load_onnx(repo, file, sha=None):
    import onnx
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo, file)
    if sha is not None:
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != sha:
            raise RuntimeError(f"{file} SHA-256 불일치: {h}")
    return onnx.load(p)


def nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    order, keep = scores.argsort()[::-1], []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = order[1:][inter / (area[i] + area[order[1:]] - inter + 1e-9) <= thr]
    return keep


class SCRFD:
    """출력 9개 = 점수·박스·랜드마크 × stride(8,16,32), 위치당 앵커 2개 (insightface scrfd.py 와 같은 디코딩)."""
    STRIDES, N_ANCHOR = (8, 16, 32), 2

    def __init__(self, device="cuda"):
        from onnx2torch import convert

        self.net = convert(_load_onnx(DET_REPO, DET_FILE)).to(device).eval()
        self.device = device

    @torch.no_grad()
    def detect(self, bgr: np.ndarray, scale: float = 2.0, thr: float = 0.3, nms_thr: float = 0.4):
        """반환: boxes (N,4) xyxy, scores (N,), kps (N,5,2) — 원본 좌표."""
        img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else bgr
        h, w = img.shape[:2]
        H, W = -(-h // 32) * 32, -(-w // 32) * 32
        pad = np.zeros((H, W, 3), np.float32)
        pad[:h, :w] = img[:, :, ::-1]
        x = torch.from_numpy((pad - 127.5) / 128.0).permute(2, 0, 1)[None].to(self.device)
        outs = [o.float().cpu().numpy() for o in self.net(x)]
        B, S, K = [], [], []
        for i, st in enumerate(self.STRIDES):
            sc, bb, kp = outs[i].reshape(-1), outs[i + 3].reshape(-1, 4) * st, outs[i + 6].reshape(-1, 10) * st
            gy, gx = np.mgrid[0:H // st, 0:W // st]
            c = np.repeat(np.stack([gx, gy], -1).reshape(-1, 2) * st, self.N_ANCHOR, 0).astype(np.float32)
            m = sc >= thr
            c, bb, kp = c[m], bb[m], kp[m].reshape(-1, 5, 2)
            B.append(np.concatenate([c - bb[:, :2], c + bb[:, 2:]], 1))
            K.append(c[:, None] + kp)
            S.append(sc[m])
        B, S, K = np.concatenate(B), np.concatenate(S), np.concatenate(K)
        keep = nms(B, S, nms_thr)
        return B[keep] / scale, S[keep], K[keep] / scale


class InSwapper:
    def __init__(self, device="cuda"):
        import onnx.numpy_helper as nh
        from onnx2torch import convert

        m = _load_onnx(SWAP_REPO, SWAP_FILE, SWAP_SHA256)
        self.emap = nh.to_array(m.graph.initializer[-1]).astype(np.float32)     # (512,512) ArcFace → 스왑 잠재
        self.net = convert(m).to(device).eval()
        self.device = device
        self.arcface = ArcFace()
        self.detail_match = False          # 66 에서 켠다 (_match_detail)

    def embed(self, bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        M, _ = cv2.estimateAffinePartial2D(np.float32(kps), ARCFACE_DST, method=cv2.LMEDS)
        return self.arcface([cv2.warpAffine(bgr, M, (112, 112), borderValue=0)])[0]

    @torch.no_grad()
    def swap(self, frame: np.ndarray, kps: np.ndarray, src_emb: np.ndarray, color_match: bool = True):
        """frame 의 얼굴(kps)을 src_emb 신원으로 바꾼 프레임을 돌려준다. (out, 붙인 영역 마스크)"""
        M, _ = cv2.estimateAffinePartial2D(np.float32(kps), SWAP_DST, method=cv2.LMEDS)
        aimg = cv2.warpAffine(frame, M, (128, 128), flags=cv2.INTER_CUBIC, borderValue=0)
        lat = src_emb.reshape(1, -1) @ self.emap
        lat /= np.linalg.norm(lat)
        x = torch.from_numpy(aimg[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        y = self.net(x.to(self.device), torch.from_numpy(lat).to(self.device))
        fake = np.clip(y[0].permute(1, 2, 0).float().cpu().numpy() * 255, 0, 255)[:, :, ::-1].astype(np.uint8)

        # 마스크는 128 정렬 공간에서 만든다 — 얼굴이 19px 라 프레임 공간에서 침식·블러하면 사라진다
        # 아래는 턱 모양 타원(끝 ≈116) — 128 바닥까지 사각형으로 붙이면 정렬 밖 검은 테두리가 섞여 턱 밑
        # 점선이 생기고, 아래 모서리가 옷깃·목 색까지 바꿨다
        mask = np.zeros((128, 128), np.float32)
        cv2.rectangle(mask, (14, 10), (113, 88), 1.0, -1)
        cv2.ellipse(mask, (64, 88), (50, 28), 0, 0, 180, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 5)
        if color_match:
            # 볼 피부만 기준으로 맞춘다 — 모자·배경까지 넣고 맞추면 피부톤이 원래 사람(목·손)과 어긋났다
            fake = _match_lab(fake, aimg, CHEEK > 0)
        # 128 → 원래 얼굴 크기로 줄어드는 비율만큼 먼저 흐려 축소 계단(앨리어싱)을 막는다
        shrink = np.sqrt(abs(np.linalg.det(M[:, :2])))           # 프레임 1px 당 정렬 공간 px
        if shrink > 1.5:
            fake = cv2.GaussianBlur(fake, (0, 0), 0.45 * shrink)
        IM = cv2.invertAffineTransform(M)
        h, w = frame.shape[:2]
        fw = cv2.warpAffine(fake, IM, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        mw = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        out = fw * mw + frame.astype(np.float32) * (1 - mw)
        if self.detail_match:
            out = _match_detail(out, frame.astype(np.float32), mw)
        return np.clip(out, 0, 255).astype(np.uint8), mw[..., 0]


def _match_detail(out: np.ndarray, frame: np.ndarray, mw: np.ndarray, k_max: float = 2.0) -> np.ndarray:
    """스왑 얼굴의 고주파(σ=1 하이패스) 세기를 원래 얼굴에 맞춘다 — 128 출력이 40px 이상 얼굴에서
    주변보다 뭉개져 붙여 넣은 티가 났다(회의실 사진). 원래보다 선명하게 만들지는 않는다(k ≥ 1)."""
    ys, xs = np.where(mw[..., 0] > 0.5)
    if len(ys) < 16:
        return out
    pad = 4
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    o, f, m = out[y0:y1, x0:x1], frame[y0:y1, x0:x1], mw[y0:y1, x0:x1]
    hp_o = o - cv2.GaussianBlur(o, (0, 0), 1.0)
    hp_f = f - cv2.GaussianBlur(f, (0, 0), 1.0)
    sel = m[..., 0] > 0.5
    k = float(np.clip(hp_f[sel].std() / (hp_o[sel].std() + 1e-3), 1.0, k_max))
    out = out.copy()
    out[y0:y1, x0:x1] = o + (k - 1.0) * hp_o * m
    return out


def _match_lab(fake: np.ndarray, ref: np.ndarray, m: np.ndarray) -> np.ndarray:
    """스왑 얼굴의 Lab 평균·대비를 원래 얼굴에 맞춘다 (같은 자리·같은 조명이어야 하므로)."""
    f = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        fm, fs = f[..., c][m].mean(), f[..., c][m].std() + 1e-3
        rm, rs = r[..., c][m].mean(), r[..., c][m].std() + 1e-3
        k = np.clip(rs / fs, 0.6, 1.4) if c == 0 else 1.0
        f[..., c] = (f[..., c] - fm) * k + rm
    return cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
