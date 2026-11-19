#!/usr/bin/env python3
"""Inspect and render Fig. 11 from the original per-image score archives."""
import argparse
import json
from pathlib import Path
import numpy as np

from generate_experiment_figures import Artwork, BLUE, TEAL, RED, GRAY, CONDITIONS
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D


VARIANTS = (
    ("Ours (QCS+semantic, condition-calibrated)",
     "output/final_results/{c}_test_seed0.json", "combined_score", BLUE, "o", "-"),
    ("Ours (QCS+semantic, unified scoring)",
     "output/unified_{c}_test.json", "combined_score", TEAL, "s", "--"),
    ("Ours (three-branch)",
     "output/{c}_three_branch_test.json", "three_branch_score", RED, "D", "-."),
)

# Deliberately include a broad interval, rather than an isolated crossing.
# Full 0--1 curves remain visible in all six main panels.
ZOOMS = {
    ("rain", "risk"): ((.20, .55), (.02, .26), [.05, .15, .25]),
    ("snow", "risk"): ((.30, .60), (.30, .37), [.31, .34, .37]),
    ("inpainting", "risk"): ((.10, .40), (.20, .36), [.20, .28, .36]),
    ("rain", "recall"): ((.20, .55), (.63, 1.00), [.65, .80, .95]),
    ("snow", "recall"): ((.30, .60), (.46, .75), [.50, .60, .70]),
    ("inpainting", "recall"): ((.55, .80), (.25, .58), [.30, .42, .54]),
}


def load_curves(artwork):
    curves = {}
    for condition in CONDITIONS:
        curves[condition] = []
        reference = None
        for name, pattern, field, color, marker, style in VARIANTS:
            source = pattern.format(c=condition)
            data, keys, labels, scores, _ = artwork.rows(source, field)
            identity = (keys, labels.tolist())
            if reference is not None and reference != identity:
                raise ValueError(f"Rejection identity mismatch: {condition}")
            reference = identity
            unknown = labels == 2
            order = np.argsort(scores, kind="stable")
            ordered = scores[order]
            endpoints = np.r_[np.flatnonzero(np.diff(ordered) != 0) + 1, len(scores)]
            accepted_unknown = np.cumsum(unknown[order])[endpoints - 1]
            coverage = endpoints / len(scores)
            risk = accepted_unknown / endpoints
            recall = 1 - accepted_unknown / unknown.sum()
            saved = data["metrics"]["conformal"]
            accepted = scores < saved["threshold"]
            point = np.array([accepted.mean(), unknown[accepted].mean(),
                              (~accepted & unknown).sum() / unknown.sum()])
            expected = [saved[key] for key in ("coverage", "selective_risk", "unknown_recall")]
            if not np.allclose(point, expected, atol=1e-6):
                raise ValueError(f"Stored rejection point mismatch: {condition}/{name}")
            curves[condition].append(dict(name=name, color=color, marker=marker,
                                           style=style, coverage=coverage,
                                           risk=risk, recall=recall, point=point,
                                           threshold=saved["threshold"], source=source))
    return curves


def step_values(curve, metric, grid):
    """Evaluate the plotted post-step function, without interpolation."""
    indices = np.searchsorted(curve["coverage"], grid, side="right") - 1
    if np.any(indices < 0):
        raise ValueError("Coverage below the first attainable nonempty acceptance set")
    return curve[metric][indices]


def inspect_windows(curves):
    candidates = ((.10, .40), (.20, .55), (.30, .60), (.40, .70), (.55, .80))
    result = {}
    for condition in CONDITIONS:
        result[condition] = {}
        for metric in ("risk", "recall"):
            result[condition][metric] = []
            for left, right in candidates:
                grid = np.arange(np.ceil(left * 422), np.floor(right * 422) + 1) / 422
                values = np.array([step_values(c, metric, grid) for c in curves[condition]])
                spread = np.ptp(values, axis=0)
                result[condition][metric].append(dict(
                    coverage=[left, right], ymin=float(values.min()), ymax=float(values.max()),
                    mean_spread=float(spread.mean()), max_spread=float(spread.max())))
    return result


def render_figure(artwork, curves=None):
    if curves is None:
        curves = load_curves(artwork)
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.25))
    records, coordinates, zoom_audit = [], [], []
    for col, condition in enumerate(CONDITIONS):
        for row, metric in enumerate(("risk", "recall")):
            ax = axes[row, col]
            for curve in curves[condition]:
                ax.step(curve["coverage"], curve[metric], where="post",
                        color=curve["color"], linestyle=curve["style"], lw=1.25,
                        label=curve["name"], zorder=3)
                point = curve["point"]
                ax.scatter(point[0], point[row + 1], marker=curve["marker"],
                           color=curve["color"], edgecolors="white", linewidth=.6,
                           s=26, zorder=5)
            if row == 0:
                ax.axhline(163 / 422, color=GRAY, ls=":", lw=.8, zorder=1)
            ax.set(xlim=(0, 1.01), ylim=(-.025, 1.025),
                   xticks=[0, .5, 1], yticks=[0, .5, 1],
                   xlabel="Acceptance coverage")
            ax.grid(True)
            ax.set_title(f"({chr(97 + row * 3 + col)}) {condition.title()}",
                         loc="left", fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel("Selective risk" if row == 0 else "Unknown recall")
            else:
                ax.tick_params(labelleft=False)

            # Place insets in empty areas: above the risk curves and below
            # the recall curves. Neither changes the data in the main axes.
            position = [.37, .52, .59, .42] if row == 0 else [.14, .09, .51, .42]
            inset = ax.inset_axes(position, zorder=8)
            xlim, ylim, yticks = ZOOMS[condition, metric]
            for curve in curves[condition]:
                # Plot the identical full arrays and clip to the local limits.
                # In particular, do not interpolate, smooth, or rescale scores.
                inset.step(curve["coverage"], curve[metric], where="post",
                           color=curve["color"], linestyle=curve["style"], lw=1.0,
                           zorder=3)
            inset.set(xlim=xlim, ylim=ylim, yticks=yticks)
            inset.set_xticks([xlim[0], (xlim[0] + xlim[1]) / 2, xlim[1]])
            inset.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            inset.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            inset.tick_params(labelsize=7, length=2, width=.5, pad=1.4)
            inset.grid(True, alpha=.18, lw=.4)
            inset.set_title("Zoom", fontsize=7, fontweight="normal", pad=2)
            inset.set_facecolor("white")
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color(GRAY)
                spine.set_linewidth(.55)

            indicator = ax.indicate_inset_zoom(inset, edgecolor=GRAY, alpha=.8,
                                               linewidth=.65, zorder=2)
            indicator.rectangle.set_linestyle((0, (3, 2)))
            for connector in indicator.connectors:
                connector.set_linestyle((0, (3, 2)))
                connector.set_linewidth(.55)

            grid = np.arange(np.ceil(xlim[0] * 422), np.floor(xlim[1] * 422) + 1) / 422
            local = np.array([step_values(c, metric, grid) for c in curves[condition]])
            # Assert that the magnification never clips a curve in its interval.
            if local.min() < ylim[0] or local.max() > ylim[1]:
                raise ValueError(f"Inset clips observations: {condition}/{metric}")
            zoom_audit.append(dict(panel=chr(97 + row * 3 + col), condition=condition,
                                   metric=metric, coverage_limits=xlim, metric_limits=ylim,
                                   mean_curve_spread=float(np.ptp(local, axis=0).mean()),
                                   maximum_curve_spread=float(np.ptp(local, axis=0).max()),
                                   inset_position_axes_fraction=position))

        for curve in curves[condition]:
            records.append([condition, curve["name"], curve["threshold"], *curve["point"]])
            coordinates.extend([condition, curve["name"], x, risk, recall]
                               for x, risk, recall in zip(curve["coverage"],
                                                          curve["risk"], curve["recall"]))

    handles = [Line2D([], [], color=v[3], ls=v[5], marker=v[4], ms=4,
                      label=v[0]) for v in VARIANTS]
    fig.legend(handles=handles, loc="lower center", ncol=1,
               bbox_to_anchor=(.54, .005), handlelength=2.5, labelspacing=.28,
               fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, top=.955, bottom=.205,
                        hspace=.52, wspace=.27)
    fig.canvas.draw()
    # Audit the exported axes, as well as the underlying score identities.
    artwork.audit["fig11_zoom_intervals"] = zoom_audit
    artwork.audit["checks"].append(
        "Fig11: all 9 archived operating points reproduced; score < threshold; "
        "ties kept together; exact post-step curves; every inset contains all local observations.")
    artwork.csv(11, "calibrated_operating_points.csv",
                ["condition", "configuration", "threshold", "coverage",
                 "selective_risk", "unknown_recall"], records)
    artwork.csv(11, "threshold_scan_coordinates.csv",
                ["condition", "configuration", "coverage", "selective_risk", "unknown_recall"],
                coordinates)
    artwork.save(fig, 11)
    (artwork.out / "figure11" / "inset_audit.json").write_text(
        json.dumps(artwork.audit, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    artwork = Artwork(args)
    curves = load_curves(artwork)
    if args.analyze_only:
        print(json.dumps(inspect_windows(curves), indent=2))
    else:
        render_figure(artwork, curves)


if __name__ == "__main__":
    main()
