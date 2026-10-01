"""8차(10명) — 결과표·그림 생성.

출력
  reports/eval_face10.md                      결과표 (시드 평균 ± 범위)
  reports/figures/fig20_face10_results.png    세 평가 조건별 4개 모델 비교 (small multiples)
  reports/figures/fig21_face10_synth_samples.png  AI 인파 합성 확대본 vs 원본 사진
  reports/figures/fig22_face10_find_demo.png  find_person 시연 (baseline vs real+synth)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths, viz  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

F10 = paths.DATASET / "faces10"
ARMS = [("baseline", "baseline (사전학습)"), ("real_deg", "실사진+열화"),
        ("synth_cctv", "AI 인파만"), ("real+synth", "실사진+AI 인파")]
# dataviz 참조 팔레트 categorical 1~4 (validate_palette.js light PASS, 대비 WARN → 값 직접 표기)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
METRICS = [("test_lowres_nat", "dir@far1%", "원래부터 저해상도인 실사진"),
           ("crowd", "find_rate@far1%", "AI 인파 속 찾기 (검출 포함)"),
           ("test_hq", "dir@far1%", "고화질 실사진")]


def agg(res, arm, m, k):
    v = [r[m][k] for key, r in res.items() if key.split("/")[0] == arm]
    return float(np.mean(v)), v


def main() -> int:
    res = json.loads((paths.REPORTS / "eval_face10.json").read_text(encoding="utf-8"))
    old_p = paths.REPORTS / "eval_face10_v1_no_crowd_neg.json"
    old = json.loads(old_p.read_text(encoding="utf-8")) if old_p.exists() else {}

    # --- 표
    lines = ["| 모델 | 고화질 실사진 | 자연 저해상도 실사진 | 열화 대조군(test_hq+Albu) | AI 인파 속 찾기 | 인파 top-1(검출된 타깃) |",
             "|---|---|---|---|---|---|"]
    for arm, lab in ARMS:
        cells = []
        for m, k in (("test_hq", "dir@far1%"), ("test_lowres_nat", "dir@far1%"), ("test_hq_albudeg", "dir@far1%"),
                     ("crowd", "find_rate@far1%"), ("crowd", "top1_detected")):
            mu, v = agg(res, arm, m, k)
            cells.append(f"{mu:.3f}" + (f" ({min(v):.3f}–{max(v):.3f})" if len(v) > 1 else ""))
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    if old:
        lines += ["", "하드 네거티브(배경 군중) 추가 전 v1 — 같은 지표:", "",
                  "| 모델 | 자연 저해상도 | AI 인파 속 찾기 | 인파 top-1 |", "|---|---|---|---|"]
        for arm, lab in ARMS[2:]:
            c = [agg(old, arm, m, k)[0] for m, k in (("test_lowres_nat", "dir@far1%"), ("crowd", "find_rate@far1%"),
                                                     ("crowd", "top1_detected"))]
            lines.append(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in c) + " |")
    (paths.REPORTS / "eval_face10.md").write_text(
        "# 8차(10명) 결과 — 모르는 사람 오인율(FAR) 1% 기준 정답률\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # --- fig20 small multiples (한 축 = 한 지표, 같은 0~1 스케일)
    viz.use_theme()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    x = np.arange(len(ARMS))
    for ax, (m, k, title) in zip(axes, METRICS):
        for i, (arm, lab) in enumerate(ARMS):
            mu, v = agg(res, arm, m, k)
            ax.bar(i, mu, width=0.62, color=COLORS[i], label=lab, zorder=2)
            ax.scatter([i] * len(v), v, s=12, color=viz.INK, zorder=3, linewidths=0)
            ax.text(i, max([mu] + v) + 0.03, f"{mu:.2f}", ha="center", va="bottom", fontsize=8, color=viz.INK)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, ["" for _ in ARMS])
        ax.set_ylim(0, 1.08)
        viz.finish(ax, xgrid=False, ygrid=True)
    axes[0].set_ylabel("정답률 @ 모르는 사람 오인율 1%")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.5, -0.04, "막대 = 2시드 평균(baseline 1회), 점 = 시드별 값. 평가 사진·사람·장면은 모두 학습에 쓰지 않음.",
             ha="center", fontsize=8, color=viz.MUTED)
    fig.savefig(paths.FIGURES / "fig20_face10_results.png")
    plt.close(fig)

    # --- fig21 합성 샘플: 프레임 확대 vs 생성 조건 원본 사진
    frames = [json.loads(l) for l in open(F10 / "cctv_frames" / "train" / "frames.jsonl", encoding="utf-8")]
    rows_m = {r.split(",")[0]: r for r in []}
    import csv
    man = list(csv.DictReader(open(F10 / "manifest.csv", encoding="utf-8")))
    tiles = []
    rng = np.random.default_rng(3)
    for fi in rng.permutation(len(frames)):
        fr = frames[fi]
        im = cv2.imread(str(F10 / "cctv_frames" / "train" / fr["image"]))
        for fc in fr["faces"]:
            if fc["cls"] >= 10 or len(tiles) >= 10:
                continue
            b = fc["box"]
            s = b[2] - b[0]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            z = im[int(max(cy - 2 * s, 0)):int(cy + 3.2 * s), int(max(cx - 2.5 * s, 0)):int(cx + 2.5 * s)]
            enroll = [r for r in man if int(r["cls"]) == fc["cls"] and r["split"] == "enroll"][0]
            src = cv2.imread(str(F10 / enroll["path"]))
            tiles.append(np.hstack([cv2.resize(z, (190, 210)), cv2.resize(src, (190, 210))]))
        if len(tiles) >= 10:
            break
    sheet = np.vstack([np.hstack(tiles[:5]), np.hstack(tiles[5:10])])
    cv2.imwrite(str(paths.FIGURES / "fig21_face10_synth_samples.png"), sheet)

    # --- fig22 find_person 시연
    demo = paths.ROOT / "runs" / "find_person"
    pair = [demo / "demo_baseline" / "test_0018.jpg", demo / "demo_real_synth" / "test_0018.jpg"]
    if all(p.exists() for p in pair):
        ims = [cv2.resize(cv2.imread(str(p)), (960, 640)) for p in pair]
        for im, t in zip(ims, ["baseline", "real+synth (s0)"]):
            cv2.putText(im, t, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.imwrite(str(paths.FIGURES / "fig22_face10_find_demo.png"), np.vstack(ims))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
