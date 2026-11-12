"""Static figure of the plate-sorting GUI for the paper (two example states).

Run from the project folder in the ROS env (has matplotlib):
    python docs/figures/make_gui_figure.py
Writes docs/figures/sort_gui.pdf and docs/figures/sort_gui.png (300 dpi).
Both examples are real captures whose prediction matches the visible plate state.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GREEN, RED = "#3cb44b", "#dc2828"
PANEL_BG, TITLE_BG, TEXT, MUTED = "#1e1e1e", "#3a3a3a", "#ebebeb", "#a8a8a8"

EXAMPLES = [
    dict(image="captures/2026-10-09/plate_21.png", label="dirty", score=0.843,
         counts=(0, 1), step="Placing in dishwasher rack"),
    dict(image="captures/2026-10-09/plate_23.png", label="clean", score=0.465,
         counts=(1, 1), step="Stacking plate on table"),
]


def draw_window(fig, rect, example):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 100, 56, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=PANEL_BG, edgecolor="#555555", linewidth=0.8))
    ax.add_patch(Rectangle((0, 51), 100, 5, facecolor=TITLE_BG, edgecolor="none"))
    ax.text(2, 53.5, "Plate sorter", color=TEXT, fontsize=7, va="center", weight="bold")
    for i, colour in enumerate(("#e0443e", "#dea123", "#1aab29")):
        ax.add_patch(Circle((91.5 + 3 * i, 53.5), 0.9, color=colour))

    # Camera view (4:3) on the left.
    image = plt.imread(str(ROOT / example["image"]))
    cam = fig.add_axes([rect[0] + rect[2] * 0.015, rect[1] + rect[3] * 0.045,
                        rect[2] * 0.6, rect[3] * 0.85])
    cam.imshow(image, aspect="auto")
    cam.set_xticks([])
    cam.set_yticks([])
    for spine in cam.spines.values():
        spine.set_edgecolor("#555555")
    cam.text(0.02, 0.96, "Live camera", transform=cam.transAxes, fontsize=5.5, color="white",
             va="top", bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=1.5))

    colour = GREEN if example["label"] == "clean" else RED
    x = 65.5
    ax.add_patch(Circle((x + 7, 40), 6, facecolor=colour, edgecolor="#dddddd", linewidth=0.8))
    ax.text(x + 15.5, 40, example["label"].upper(), color=colour, fontsize=10, weight="bold", va="center")
    ax.text(x, 30.5, f"Dirty score: {example['score']:.3f}", color=TEXT, fontsize=6)
    ax.text(x, 24.5, "Mixed rack count", color=TEXT, fontsize=6.5, weight="bold")
    clean, dirty = example["counts"]
    ax.text(x, 20, f"Clean: {clean}", color=GREEN, fontsize=6.5, weight="bold")
    ax.text(x, 16, f"Dirty: {dirty}", color=RED, fontsize=6.5, weight="bold")
    ax.text(x, 12, f"Total: {clean + dirty}", color=TEXT, fontsize=6.5)
    ax.text(x, 6.5, example["step"], color="#ffbe00", fontsize=5.2, wrap=True)
    ax.text(x, 2.5, "Space: start   R: reset   Q: quit", color=MUTED, fontsize=4.6)


def main():
    fig = plt.figure(figsize=(7.16, 2.15))
    for i, (example, tag) in enumerate(zip(EXAMPLES, "ab")):
        left = 0.01 + i * 0.5
        draw_window(fig, [left, 0.1, 0.48, 0.88], example)
        fig.text(left + 0.24, 0.025, f"({tag}) {example['label'].capitalize()} plate detected",
                 ha="center", fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"sort_gui.{suffix}", dpi=300)
    print(f"Saved {HERE / 'sort_gui.pdf'} and .png")


if __name__ == "__main__":
    main()
