"""COCO 2017 클래스 분포 전수 집계 → 희소 클래스 선정.

사전학습 YOLO(COCO 80 클래스)가 "어떤 클래스에 데이터가 부족한가"를
정량으로 답하는 단계. train2017 인스턴스 어노테이션 전체를 읽어

  - 클래스별 인스턴스 수 / 이미지 수 / 점유율 / 순위
  - 이미지당 인스턴스 수
  - COCO 공식 크기 구간 비율·절대수 (small <32^2, medium 32^2~96^2, large >96^2)
  - 상대 bbox 면적 중앙값
  - iscrowd 비율

를 뽑고, 아래 3중 기준으로 타깃 희소 클래스를 고른다.

  S1 (절대 희소성)   인스턴스 수가 80클래스 분포의 하위 사분위(Q1) 이하
  S2 (저화질 취약)   small 구간 인스턴스의 '절대 개수'가 하위 사분위 이하
  S3 (도메인 관련)   야생동물 활동 영상 / 실외 감시 도메인에 실제로 등장

S2 를 비율이 아니라 절대수로 두는 이유: 저화질·원거리 영상에서 객체는 작은
스케일로 나타나므로, 검출기가 그 클래스를 '작게' 본 학습 사례의 수가 곧 저화질
성능의 상한이 된다. 비율이 낮아도 총량이 많으면(person) 문제가 없고, 비율이
높아도 총량이 적으면(hair drier) 문제가 된다. 두 축을 모두 보려면 절대수가 맞다.

산출물
  reports/coco_class_stats.csv
  reports/coco_class_stats.md
  reports/rare_class_selection.json
  reports/figures/fig01_class_distribution.png
  reports/figures/fig02_animal_classes.png
  reports/figures/fig03_scarcity_vs_smallratio.png
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rareobj import paths, viz  # noqa: E402

# Windows 콘솔 기본 코드페이지(cp949)에서 한글/기호 출력이 깨지는 것 방지
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

SMALL = 32**2
MEDIUM = 96**2

# 야생동물 활동 영상 / 실외 저화질 감시 도메인에서 실제로 관측되는 클래스
# (5/14 공유된 야생동물 영상 자료 + COCO 클래스 리스트 대조)
DOMAIN_WILDLIFE = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}


def load(split: str) -> dict:
    p = paths.COCO_ANN / f"instances_{split}2017.json"
    if not p.exists():
        raise SystemExit(f"어노테이션 없음: {p}\n먼저 scripts/00_download_coco.py 실행")
    print(f"[load] {p.name} ({p.stat().st_size/1e6:.0f} MB) …")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def aggregate(ann: dict) -> tuple[list[dict], dict]:
    cats = {c["id"]: c for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}

    n_inst: dict[int, int] = defaultdict(int)
    n_crowd: dict[int, int] = defaultdict(int)
    img_ids: dict[int, set[int]] = defaultdict(set)
    sizes: dict[int, list[int]] = defaultdict(list)   # 0 small / 1 medium / 2 large
    rel_area: dict[int, list[float]] = defaultdict(list)

    for a in ann["annotations"]:
        cid = a["category_id"]
        n_inst[cid] += 1
        n_crowd[cid] += int(a.get("iscrowd", 0))
        img_ids[cid].add(a["image_id"])
        area = float(a["area"])
        sizes[cid].append(0 if area < SMALL else (1 if area < MEDIUM else 2))
        im = imgs[a["image_id"]]
        rel_area[cid].append(area / (im["width"] * im["height"]))

    total = sum(n_inst.values())
    rows = []
    for cid, cat in cats.items():
        n = n_inst[cid]
        sz = np.asarray(sizes[cid]) if n else np.zeros(0)
        rows.append(
            {
                "class": cat["name"],
                "supercategory": cat["supercategory"],
                "coco_id": cid,
                "n_inst": n,
                "n_img": len(img_ids[cid]),
                "share_pct": 100 * n / total,
                "inst_per_img": n / max(len(img_ids[cid]), 1),
                "small_ratio": float((sz == 0).mean()) if n else 0.0,
                "medium_ratio": float((sz == 1).mean()) if n else 0.0,
                "large_ratio": float((sz == 2).mean()) if n else 0.0,
                "n_small": int((sz == 0).sum()) if n else 0,
                "n_medium": int((sz == 1).sum()) if n else 0,
                "n_large": int((sz == 2).sum()) if n else 0,
                "rel_area_median": float(np.median(rel_area[cid])) if n else 0.0,
                "crowd_ratio": n_crowd[cid] / max(n, 1),
            }
        )

    rows.sort(key=lambda r: r["n_inst"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["imbalance_vs_top"] = rows[0]["n_inst"] / r["n_inst"] if r["n_inst"] else float("inf")

    meta = {
        "n_images": len(ann["images"]),
        "n_annotations": len(ann["annotations"]),
        "n_classes": len(cats),
        "total_instances": total,
    }
    return rows, meta


def select_rare(rows: list[dict]) -> dict:
    counts = np.array([r["n_inst"] for r in rows], float)
    n_small = np.array([r["n_small"] for r in rows], float)
    q1_inst = float(np.percentile(counts, 25))
    q1_small = float(np.percentile(n_small, 25))

    for r in rows:
        r["S1_scarce"] = r["n_inst"] <= q1_inst
        r["S2_small_scarce"] = r["n_small"] <= q1_small
        r["S3_domain"] = r["class"] in DOMAIN_WILDLIFE
        r["n_criteria"] = int(r["S1_scarce"]) + int(r["S2_small_scarce"]) + int(r["S3_domain"])

    triple = [r for r in rows if r["n_criteria"] == 3]
    animals = sorted((r for r in rows if r["class"] in DOMAIN_WILDLIFE),
                     key=lambda r: r["n_small"])

    keys = ("class", "n_inst", "n_img", "rank", "n_small", "small_ratio", "share_pct")
    return {
        "thresholds": {
            "q1_instance_count": q1_inst,
            "q1_n_small": q1_small,
            "total_instances": int(counts.sum()),
        },
        "S1_scarce_classes": [r["class"] for r in rows if r["S1_scarce"]],
        "S2_small_scarce_classes": [r["class"] for r in rows if r["S2_small_scarce"]],
        "all_three_criteria": [{k: r[k] for k in keys} for r in triple],
        "wildlife_ranked_by_n_small": [{k: r[k] for k in keys} for r in animals],
    }


# ----------------------------------------------------------------- figures
def fig_distribution(rows: list[dict], q1: float, targets: set[str]) -> Path:
    """80 클래스 랭킹. 인스턴스 수가 3자릿수~6자릿수를 넘나들어 막대(선형)로는
    희소 클래스가 아예 안 보인다 → log 축 + 점 플롯(lollipop)이 올바른 형태."""
    viz.use_theme()
    ordered = rows[::-1]  # 아래에서 위로 증가
    y = np.arange(len(ordered))
    n = np.array([r["n_inst"] for r in ordered], float)
    is_t = np.array([r["class"] in targets for r in ordered])

    fig, ax = plt.subplots(figsize=(7.2, 11.0))
    ax.hlines(y[~is_t], 1, n[~is_t], color=viz.BLUE_LT, lw=1.4)
    ax.plot(n[~is_t], y[~is_t], "o", ms=4.5, color=viz.BLUE, label="COCO 80 클래스")
    if is_t.any():
        ax.hlines(y[is_t], 1, n[is_t], color=viz.CRITICAL, lw=1.8)
        ax.plot(n[is_t], y[is_t], "o", ms=6.5, color=viz.CRITICAL, label="선정된 희소 클래스")

    ax.axvline(q1, color=viz.MUTED, ls=":", lw=1.2)
    ax.text(q1, len(ordered) - 0.5, f"  Q1 = {q1:,.0f}", color=viz.INK_2, fontsize=8, va="top")

    ax.set_xscale("log")
    ax.set_xlim(150, 3.5e5)
    ax.set_ylim(-1, len(ordered))
    ax.set_yticks(y, [r["class"] for r in ordered], fontsize=6.6)
    ax.set_xlabel("인스턴스 수 (train2017, log scale)")
    ax.set_title("COCO train2017 클래스별 인스턴스 수", loc="left", pad=12)

    for i in np.where(is_t)[0]:
        ax.annotate(f"{int(n[i]):,}", (n[i], y[i]), xytext=(8, 0),
                    textcoords="offset points", va="center",
                    fontsize=7.5, color=viz.INK, fontweight="bold")
    ax.legend(loc="lower right")
    viz.finish(ax)

    out = paths.FIGURES / "fig01_class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_animals(rows: list[dict], targets: set[str]) -> Path:
    """같은 클래스 순서를 공유하는 2패널. 축 스케일이 서로 달라 한 그림에 겹치면
    안 되므로(이중축 금지) small multiples 로 나눈다."""
    viz.use_theme()
    animals = sorted((r for r in rows if r["class"] in DOMAIN_WILDLIFE),
                     key=lambda r: r["n_inst"])
    y = np.arange(len(animals))
    labels = [r["class"] for r in animals]
    colors = [viz.CRITICAL if r["class"] in targets else viz.BLUE for r in animals]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    panels = [
        ("n_inst", "전체 인스턴스 수", axes[0]),
        ("n_small", "small 구간 인스턴스 수  (area < 32² px)", axes[1]),
    ]
    for key, xlabel, ax in panels:
        v = np.array([r[key] for r in animals], float)
        ax.barh(y, v, height=0.62, color=colors)
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, v.max() * 1.24)
        for i, val in enumerate(v):
            ax.annotate(f"{int(val):,}", (val, i), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=viz.INK,
                        fontweight="bold" if colors[i] == viz.CRITICAL else "normal")
        viz.finish(ax)
    axes[1].set_yticklabels([])

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=7, color=viz.CRITICAL, label="선정된 희소 클래스"),
        plt.Line2D([], [], marker="s", ls="", ms=7, color=viz.BLUE, label="그 외 동물 클래스"),
    ]
    axes[1].legend(handles=handles, loc="lower right")
    fig.suptitle("야생동물 도메인 클래스 — 총량 대비 저해상도 커버리지",
                 x=0.055, ha="left", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = paths.FIGURES / "fig02_animal_classes.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_scatter(rows: list[dict], q1_inst: float, q1_small: float, targets: set[str]) -> Path:
    """두 선정 기준을 그대로 2차원 평면으로. 좌하단 음영이 S1∩S2 교집합 구간."""
    viz.use_theme()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    other = [r for r in rows if r["class"] not in targets]
    tgt = [r for r in rows if r["class"] in targets]

    ax.add_patch(plt.Rectangle((150, 8), q1_inst - 150, q1_small - 8,
                               color=viz.GRID, alpha=0.7, lw=0, zorder=0))
    ax.axvline(q1_inst, color=viz.MUTED, ls=":", lw=1.2)
    ax.axhline(q1_small, color=viz.MUTED, ls=":", lw=1.2)
    ax.text(q1_inst * 0.9, 6e4, f"S1: 인스턴스 ≤ {q1_inst:,.0f}", ha="right", va="top",
            fontsize=7.5, color=viz.INK_2, rotation=90)
    ax.text(3.2e5, q1_small * 1.12, f"S2: small 인스턴스 ≤ {q1_small:,.0f}",
            ha="right", va="bottom", fontsize=7.5, color=viz.INK_2)
    ax.text(180, 11, "S1 ∩ S2", fontsize=8.5, color=viz.INK_2, fontweight="bold")

    ax.scatter([r["n_inst"] for r in other], [max(r["n_small"], 9) for r in other],
               s=26, color=viz.BLUE, alpha=0.75, lw=0, label="COCO 80 클래스", zorder=3)
    if tgt:
        ax.scatter([r["n_inst"] for r in tgt], [max(r["n_small"], 9) for r in tgt],
                   s=110, color=viz.CRITICAL, lw=1.5, edgecolor=viz.SURFACE,
                   label="선정된 희소 클래스", zorder=5)
        for r in tgt:
            ax.annotate(f"{r['class']}\n전체 {r['n_inst']:,} / small {r['n_small']:,}",
                        (r["n_inst"], max(r["n_small"], 9)), xytext=(-13, 2),
                        textcoords="offset points", ha="right", fontsize=8.5,
                        color=viz.INK, fontweight="bold", zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(150, 3.5e5)
    ax.set_ylim(8, 1.2e5)
    ax.set_xlabel("전체 인스턴스 수 (log)")
    ax.set_ylabel("small 구간 인스턴스 수  (area < 32² px, log)")
    ax.set_title("선정 기준 S1 × S2 — 희소성과 저해상도 커버리지", loc="left", pad=10)
    ax.legend(loc="lower right")
    ax.set_axisbelow(True)
    ax.grid(True)

    out = paths.FIGURES / "fig03_scarcity_vs_smallcount.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------- writers
COLS = [
    ("rank", "순위", "{:d}"),
    ("class", "클래스", "{}"),
    ("supercategory", "상위분류", "{}"),
    ("n_inst", "인스턴스", "{:,d}"),
    ("n_img", "이미지", "{:,d}"),
    ("share_pct", "점유율%", "{:.3f}"),
    ("inst_per_img", "이미지당", "{:.2f}"),
    ("n_small", "small수", "{:,d}"),
    ("small_ratio", "small비율", "{:.3f}"),
    ("n_medium", "medium수", "{:,d}"),
    ("n_large", "large수", "{:,d}"),
    ("rel_area_median", "상대면적중앙값", "{:.5f}"),
    ("imbalance_vs_top", "1위대비배수", "{:.1f}"),
]


def write_csv(rows: list[dict], out: Path) -> None:
    import csv

    keys = [k for k, _, _ in COLS] + ["S1_scarce", "S2_small_scarce", "S3_domain", "n_criteria"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_md(rows: list[dict], meta: dict, sel: dict, out: Path) -> None:
    L = [
        "# COCO train2017 클래스 분포 전수 집계",
        "",
        f"- 이미지 {meta['n_images']:,} / 어노테이션 {meta['n_annotations']:,} "
        f"/ 클래스 {meta['n_classes']} / 인스턴스 합 {meta['total_instances']:,}",
        f"- S1 임계 (인스턴스 수 하위 사분위) Q1 = {sel['thresholds']['q1_instance_count']:,.0f}",
        f"- S2 임계 (small 인스턴스 수 하위 사분위) Q1 = {sel['thresholds']['q1_n_small']:,.0f}",
        "",
        "| " + " | ".join(h for _, h, _ in COLS) + " | S1 | S2 | S3 |",
        "|" + "---|" * (len(COLS) + 3),
    ]
    for r in rows:
        cells = [fmt.format(r[k]) for k, _, fmt in COLS]
        flags = ["✓" if r[k] else "" for k in ("S1_scarce", "S2_small_scarce", "S3_domain")]
        L.append("| " + " | ".join(cells + flags) + " |")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    paths.ensure()
    ann = load("train")
    rows, meta = aggregate(ann)
    sel = select_rare(rows)

    targets = {d["class"] for d in sel["all_three_criteria"]}
    q1_inst = sel["thresholds"]["q1_instance_count"]
    q1_small = sel["thresholds"]["q1_n_small"]

    write_csv(rows, paths.REPORTS / "coco_class_stats.csv")
    write_md(rows, meta, sel, paths.REPORTS / "coco_class_stats.md")

    sel["meta"] = meta
    sel["figures"] = [
        fig_distribution(rows, q1_inst, targets).name,
        fig_animals(rows, targets).name,
        fig_scatter(rows, q1_inst, q1_small, targets).name,
    ]
    (paths.REPORTS / "rare_class_selection.json").write_text(
        json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    out = [
        f"[meta] {meta}",
        f"[thresh] S1 Q1(n_inst)={q1_inst:,.0f}   S2 Q1(n_small)={q1_small:,.0f}",
        "",
        "[인스턴스 수 하위 10]",
    ]
    for r in rows[-10:]:
        out.append(f"  {r['rank']:>2}. {r['class']:<16} n={r['n_inst']:>7,} "
                   f"img={r['n_img']:>6,} small={r['n_small']:>6,} ({r['small_ratio']:.3f}) "
                   f"{r['supercategory']}")
    out += ["", "[small 인스턴스 수 하위 10]"]
    for r in sorted(rows, key=lambda r: r["n_small"])[:10]:
        out.append(f"  {r['class']:<16} small={r['n_small']:>6,}  n={r['n_inst']:>7,}  "
                   f"rank={r['rank']:>2}")
    out += ["", "[야생동물 도메인 — small 인스턴스 오름차순]"]
    for d in sel["wildlife_ranked_by_n_small"]:
        out.append(f"  {d['class']:<10} small={d['n_small']:>6,}  n={d['n_inst']:>7,}  "
                   f"img={d['n_img']:>6,}  rank={d['rank']:>2}")
    out += [
        "",
        f"[3중 기준 동시 충족] {sorted(targets) or '없음'}",
        f"[S1 {len(sel['S1_scarce_classes'])}개] {sel['S1_scarce_classes']}",
        f"[S2 {len(sel['S2_small_scarce_classes'])}개] {sel['S2_small_scarce_classes']}",
    ]
    text = "\n".join(out)
    (paths.REPORTS / "selection_console.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
