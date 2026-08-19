"""리포트 figure 공통 스타일.

색은 역할로만 지정한다. BLUE = 일반 계열(단일 시계열/크기), CRITICAL = 선정된
희소 클래스 강조. 강조색은 상태색이라 색 하나로 의미를 지지 않도록 legend +
직접 라벨을 항상 같이 넣는다.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- palette (light surface) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
BLUE_LT = "#9ec5f4"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"


def use_theme() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "Segoe UI", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
        }
    )


def finish(ax: plt.Axes, *, xgrid: bool = True, ygrid: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", visible=xgrid or ygrid)
    if xgrid:
        ax.grid(axis="y", visible=False)
    if ygrid:
        ax.grid(axis="y", visible=True)
