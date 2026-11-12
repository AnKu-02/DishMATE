"""Framework figure: (a) robot setup photo, (b) sorting workflow with the GUI.

Run from the project folder in the ROS env (has matplotlib):
    python docs/figures/make_framework_figure.py --setup-photo path/to/setup.jpg
Without --setup-photo, panel (a) is a labelled placeholder.
Writes docs/figures/framework.pdf and docs/figures/framework.png (300 dpi).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
INK, MUTED, EDGE = "#222222", "#666666", "#9a9a9a"
PERCEPTION, MANIPULATION, GUI = "#e3edf8", "#fbeadb", "#e6f3e6"
GREEN, RED = "#2e9e44", "#d03030"


def box(ax, x, y, w, h, title, detail=None, fill="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=fill, edgecolor=EDGE, linewidth=0.7))
    if detail:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=6.6,
                color=INK, weight="bold")
        ax.text(x + w / 2, y + h * 0.3, detail, ha="center", va="center", fontsize=5.6,
                color=MUTED, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=6.6,
                color=INK, weight="bold")


def arrow(ax, start, end, label=None, color=INK, style="-|>", dashed=False, rad=0.0, label_xy=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=7, color=color,
                                 linewidth=0.8, linestyle=(0, (3, 2)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.6)
        ax.text(lx, ly, label, ha="center", va="bottom", fontsize=5.8, color=color, weight="bold")


def setup_panel(fig, rect, photo):
    ax = fig.add_axes(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    if photo:
        ax.imshow(plt.imread(str(photo)), aspect="auto")
        for spine in ax.spines.values():
            spine.set_edgecolor(EDGE)
    else:
        ax.set_facecolor("#f2f2f2")
        for spine in ax.spines.values():
            spine.set_edgecolor(EDGE)
            spine.set_linestyle((0, (4, 3)))
        ax.text(0.5, 0.56, "Robot setup photo", ha="center", va="center", fontsize=8,
                color=MUTED, weight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.42, "MAiRA + RG6 gripper, wrist camera,\nmixed rack, dishwasher rack,\nclean-plate stack",
                ha="center", va="center", fontsize=6, color=MUTED, transform=ax.transAxes)


def workflow_panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # Lanes.
    ax.add_patch(Rectangle((0, 34), 73, 27, facecolor=PERCEPTION, edgecolor="none"))
    ax.add_patch(Rectangle((0, 1), 73, 30.5, facecolor=MANIPULATION, edgecolor="none"))
    ax.text(1.2, 59.3, "Perception", fontsize=6.4, color="#2d5f93", weight="bold", va="top")
    ax.text(1.2, 29.8, "Manipulation", fontsize=6.4, color="#a35a1c", weight="bold", va="top")

    # Perception lane: capture -> preprocess -> classify -> decision.
    box(ax, 1.5, 38, 15, 15, "Capture", "wrist camera at\nobserve pose")
    box(ax, 19.5, 38, 15, 15, "Preprocess", "plate ROI crop,\nRGB, 224×224")
    box(ax, 37.5, 38, 15, 15, "Classify", "MobileNetV2\ndirty score p")
    diamond = [(62.5, 53.5), (70.5, 45.5), (62.5, 37.5), (54.5, 45.5)]
    ax.add_patch(Polygon(diamond, closed=True, facecolor="#ffffff", edgecolor=EDGE, linewidth=0.7))
    ax.text(62.5, 45.5, "p > τ ?", ha="center", va="center", fontsize=6.6, color=INK, weight="bold")
    for x0, x1 in ((16.5, 19.5), (34.5, 37.5), (52.5, 54.5)):
        arrow(ax, (x0, 45.5), (x1, 45.5))

    # Manipulation lane runs right to left: pick & lift -> place (dirty / clean) -> return.
    box(ax, 55.5, 9, 14, 15, "Pick & lift", "RG6 close,\nlift waypoint")
    box(ax, 24, 17.5, 20, 10, "Dishwasher rack", "dirty: place, release", fill="#fde3e3")
    box(ax, 24, 4, 20, 10, "Table stack", "clean: place, release", fill="#e2f4e5")
    box(ax, 1.5, 9, 13, 15, "Return", "to observe\npose")

    arrow(ax, (62.5, 37.5), (62.5, 24))
    arrow(ax, (55.5, 19), (44, 22.5), "dirty", color=RED, label_xy=(49.8, 22.4))
    arrow(ax, (55.5, 14), (44, 9), "clean", color=GREEN, label_xy=(49.8, 12.4))
    arrow(ax, (24, 22.5), (14.5, 19))
    arrow(ax, (24, 9), (14.5, 14))
    # Next plate loop.
    arrow(ax, (8, 24), (8, 38), color=MUTED, dashed=True)
    ax.text(9.5, 31.6, "next plate", fontsize=5.6, color=MUTED, ha="left", va="center", style="italic")

    # GUI column with a thumbnail of the real interface.
    ax.add_patch(Rectangle((75, 1), 25, 60, facecolor=GUI, edgecolor="none"))
    ax.text(87.5, 59.3, "Operator GUI", fontsize=6.4, color="#2b6b34", weight="bold", ha="center", va="top")
    gui = plt.imread(str(HERE / "sort_gui.png"))
    height, width = gui.shape[:2]
    thumb = gui[: int(height * 0.9), : width // 2]
    tx = rect[0] + rect[2] * 0.765
    ty = rect[1] + rect[3] * (30 / 62)
    tax = fig.add_axes([tx, ty, rect[2] * 0.22, rect[3] * (24 / 62)])
    tax.imshow(thumb, aspect="auto")
    tax.axis("off")
    for i, line in enumerate(("live camera view", "status light: red = dirty,",
                              "green = clean", "clean / dirty counts", "current robot step")):
        ax.text(77, 26 - 4.2 * i, ("• " if i in (0, 1, 3, 4) else "   ") + line,
                fontsize=5.6, color=INK, va="center")
    arrow(ax, (70.5, 45.5), (75, 45.5), color="#2b6b34", dashed=True)
    arrow(ax, (69.5, 16.5), (75, 16.5), color="#2b6b34", dashed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup-photo", type=Path, help="photo of the robot cell for panel (a)")
    args = parser.parse_args()

    fig = plt.figure(figsize=(7.16, 2.7))
    setup_panel(fig, [0.01, 0.1, 0.27, 0.88], args.setup_photo)
    workflow_panel(fig, [0.3, 0.1, 0.69, 0.88])
    fig.text(0.145, 0.025, "(a) Experimental setup", ha="center", fontsize=8)
    fig.text(0.645, 0.025, "(b) Sorting workflow and operator interface", ha="center", fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"framework.{suffix}", dpi=300)
    print(f"Saved {HERE / 'framework.pdf'} and .png")


if __name__ == "__main__":
    main()
