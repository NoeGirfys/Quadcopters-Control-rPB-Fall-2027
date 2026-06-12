"""Overlay the MAML mass-attachment task distribution on the real Crazyflie photo.

Slide-16 figure: the per-task mass clusters are drawn on top of a top-view photo
of the drone, so the abstract "taped mass" task distribution becomes tangible.

WHY THE TWO-STEP DESIGN
-----------------------
Matching report Fig. 18 by *rotating* the photo is impossible if the photo and
the figure differ by a reflection (a mirrored / from-below shot). So clicking is
purely GEOMETRIC (you click the four motor hubs by their position in the image,
no need to know where the drone's front is), and the mapping from an image corner
to a motor number lives in CORNER_TO_MOTOR below. If the layout does not match
Fig. 18, just edit that dict (any permutation, reflections included), set
REUSE_SAVED=True and re-run -- the cached clicks make it instant.

Fig. 18 reference (shown during clicking):
    M1 = blue = front-right, M2 = orange = rear-right,
    M3 = green = rear-left,  M4 = cross  = front-left  (held-out).

Interaction: left-click place, right-click undo, Enter when a phase is done.
"""
from __future__ import annotations

import json
import os

import matplotlib
try:                                   # interactive backend for clicking
    matplotlib.use("TkAgg")
except Exception:
    pass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
PHOTO = "../Crazyflie_topwiew_presentation.png"
REF = "../Report_Latex_Overleaf/images/maml_offdirmag_map.png"   # Fig. 18
OUT = "../Report_Latex_Overleaf/images/maml_tasks_overlay.png"
CLICK_CACHE = "motor_pix.json"
REUSE_SAVED = False        # True -> load cached clicks (tweak colours/mapping fast)

# Image corners, in the order you will CLICK them (pure image geometry).
CORNERS = ["bottom-left", "top-left", "top-right", "bottom-right"]

# Which motor number sits at each image corner. EDIT THIS to match Fig. 18.
# Default assumes the drone's front is at the BOTTOM of the photo. If the overlay
# comes out mirrored w.r.t. Fig. 18, swap entries here (e.g. left<->right) and
# re-run with REUSE_SAVED=True.
CORNER_TO_MOTOR = {
    "bottom-left":  "M1",
    "top-left":     "M2",
    "top-right":    "M3",
    "bottom-right": "M4",   # held-out
}
HELD_OUT = "M4"

# Arm geometry (maml_lib/config.py): L = 0.0397 m, motors at (+/-a, +/-a) in body.
A_CM = 0.0397 / np.sqrt(2.0) * 100.0          # ~= 2.807 cm

# Task distribution (tasks_offdirmag.yaml)
SIGMA_CM = 0.3                                  # mass_pos_sigma = 0.003 m
N_PTS = 90                                      # sampled points per task
SEED = 0
MAGNITUDES = (3.0, 6.0)
SIZE_BY_MASS = {3.0: 22.0, 6.0: 95.0}           # small = 3 g, large = 6 g

# One colour per task, all six well separated (M3's two are now distinct).
TASK_COLORS = {
    ("M1", 3.0): "#1f77b4", ("M1", 6.0): "#ff7f0e",   # blue / orange
    ("M2", 3.0): "#2ca02c", ("M2", 6.0): "#d62728",   # green / red
    ("M3", 3.0): "#9467bd", ("M3", 6.0): "#17becf",   # purple / teal
}


def collect_clicks(img: np.ndarray):
    """Two-phase pick (centres, then label positions), keyed by image corner."""
    if REUSE_SAVED and os.path.exists(CLICK_CACHE):
        with open(CLICK_CACHE) as f:
            d = json.load(f)
        return d["centers"], d["labels"]

    h, w = img.shape[:2]

    # reference window (Fig. 18 left panel) kept open during clicking
    ref_img = mpimg.imread(REF)
    ref_left = ref_img[:, : ref_img.shape[1] // 2]
    ref_fig, ref_ax = plt.subplots(figsize=(6, 6 * ref_left.shape[0] / ref_left.shape[1]))
    ref_ax.imshow(ref_left)
    ref_ax.set_title("REFERENCE - report Fig. 18\nM1 blue=FR, M2 orange=RR, "
                     "M3 green=RL, M4 cross=FL", fontsize=9)
    ref_ax.axis("off")
    ref_fig.show()
    plt.pause(0.3)

    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(
        "Phase 1/2 - click the 4 motor HUBS by image position, in order:\n"
        "1) bottom-left  2) top-left  3) top-right  4) bottom-right\n"
        "(left-click place, right-click undo, Enter when done)", fontsize=11)
    center_pts = fig.ginput(4, timeout=0)
    if len(center_pts) != 4:
        raise RuntimeError(f"Phase 1: expected 4 clicks, got {len(center_pts)}.")
    for (x, y), corner in zip(center_pts, CORNERS):
        col = "crimson" if CORNER_TO_MOTOR[corner] == HELD_OUT else "k"
        ax.plot(x, y, marker="+", color=col, ms=14, mew=2.0)
    fig.canvas.draw()

    ax.set_title(
        "Phase 2/2 - click where to PLACE each LABEL, same corner order:\n"
        "1) bottom-left  2) top-left  3) top-right  4) bottom-right\n"
        "(left-click place, right-click undo, Enter when done)", fontsize=11)
    fig.canvas.draw()
    label_pts = fig.ginput(4, timeout=0)
    if len(label_pts) != 4:
        raise RuntimeError(f"Phase 2: expected 4 clicks, got {len(label_pts)}.")
    plt.close("all")

    centers = {c: [float(x), float(y)] for c, (x, y) in zip(CORNERS, center_pts)}
    labels = {c: [float(x), float(y)] for c, (x, y) in zip(CORNERS, label_pts)}
    with open(CLICK_CACHE, "w") as f:
        json.dump({"centers": centers, "labels": labels}, f, indent=2)
    print(f"saved clicks -> {CLICK_CACHE}")
    return centers, labels


def pixel_scale(centers: dict) -> float:
    """px-per-cm from the clicked motor square (rotation-independent)."""
    pix = np.array([centers[c] for c in CORNERS], dtype=float)
    centroid = pix.mean(0)
    mean_pix_r = np.linalg.norm(pix - centroid, axis=1).mean()
    return mean_pix_r / (A_CM * np.sqrt(2.0))


def main() -> None:
    rng = np.random.default_rng(SEED)
    img = mpimg.imread(PHOTO)
    h, w = img.shape[:2]

    centers, labels = collect_clicks(img)
    sigma_px = SIGMA_CM * pixel_scale(centers)

    fig, ax = plt.subplots(figsize=(8, 8 * h / w), dpi=300)
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    for corner in CORNERS:
        motor = CORNER_TO_MOTOR[corner]
        cx, cy = centers[corner]
        is_held = motor == HELD_OUT
        if not is_held:
            # large (6 g) first so small (3 g) stays visible on top
            for mass in sorted(MAGNITUDES, reverse=True):
                pts = np.array([cx, cy]) + rng.normal(0.0, sigma_px, (N_PTS, 2))
                ax.scatter(pts[:, 0], pts[:, 1], s=SIZE_BY_MASS[mass],
                           color=TASK_COLORS[(motor, mass)], alpha=0.5,
                           edgecolors="white", linewidths=0.25, zorder=3,
                           label=f"{motor} {int(mass)}g")
        col = "crimson" if is_held else "k"
        ax.plot(cx, cy, marker="+", color=col, ms=15, mew=2.0, zorder=5)
        lx, ly = labels[corner]
        ax.annotate(motor, (lx, ly), fontsize=13, fontweight="bold",
                    color=col, ha="center", va="center", zorder=5)

    handles, lab = ax.get_legend_handles_labels()
    order = sorted(range(len(lab)), key=lambda i: lab[i])
    handles = [handles[i] for i in order]
    lab = [lab[i] for i in order]
    handles.append(Line2D([], [], marker="+", color="crimson", linestyle="None",
                          markersize=11, markeredgewidth=2.2))
    lab.append(f"{HELD_OUT} ~5 g (held-out)")
    ax.legend(handles, lab, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=9, frameon=False)

    ax.set_title("Task distribution: taped-mass position on the drone body\n"
                 "(colour = task, marker size = mass: small 3 g / large 6 g)",
                 fontsize=12, pad=8)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
