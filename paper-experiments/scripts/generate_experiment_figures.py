#!/usr/bin/env python3
"""Reproduce experimental artwork, Fig. 6--12, from archived predictions.

No model inference, synthetic observations, score inversion, or missing-file
substitution is used. Supply data and font locations for your own environment.
"""
import argparse
import csv
import hashlib
import json
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import rankdata, spearmanr

CONDITIONS = ("rain", "snow", "inpainting")
BLUE, RED, TEAL, GOLD, PURPLE = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"
INK, GRAY = "#263445", "#7A8793"
CONFIGS = ("full", "no_pair", "no_known")
CONFIG_LABELS = ("Ours (full DQCE)", "Ours w/o pair", "Ours w/o known supervision")
CONFIG_COLORS = (RED, BLUE, TEAL)
STEMS = {6: "detection_comparison", 7: "unknown_score_distributions",
         8: "dqce_ablation", 9: "open_world_scores", 10: "accuracy_efficiency",
         11: "selective_risk_coverage", 12: "fusion_analysis"}


class Artwork:
    def __init__(self, args):
        self.args = args
        self.repo = args.repo_root.resolve()
        self.out = args.figure_root.resolve()
        self.audit = {"sources": {}, "checks": [], "outputs": [],
                      "score_direction": "Archived direction; never inverted",
                      "font": "Times New Roman", "dpi": args.dpi}
        self.font = self.configure_font(args.font_dir)

    def configure_font(self, requested):
        candidates = [requested] if requested else [Path.home() / ".fonts"]
        for folder in candidates:
            if folder and (folder / "times.ttf").is_file():
                for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
                    if (folder / name).is_file():
                        font_manager.fontManager.addfont(str(folder / name))
                font = folder / "times.ttf"
                break
        else:
            prop = font_manager.FontProperties(family="Times New Roman")
            try:
                font = Path(font_manager.findfont(prop, fallback_to_default=False))
            except ValueError as exc:
                raise RuntimeError("Times New Roman is required. Set --font-dir to its real font directory.") from exc
        resolved = font_manager.FontProperties(fname=str(font)).get_name()
        if resolved != "Times New Roman":
            raise RuntimeError(f"Unexpected font family: {resolved}")
        plt.rcParams.update({
            "font.family": "Times New Roman", "font.size": 9,
            "axes.labelsize": 9, "axes.titlesize": 10,
            "xtick.labelsize": 8, "ytick.labelsize": 8,
            "legend.fontsize": 8, "legend.frameon": False,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": .65, "axes.edgecolor": INK, "text.color": INK,
            "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
            "axes.axisbelow": True, "grid.alpha": .18, "grid.linewidth": .5,
            "lines.linewidth": 1.35, "pdf.fonttype": 42, "ps.fonttype": 42,
            "savefig.facecolor": "white", "figure.facecolor": "white",
            "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
            "mathtext.cal": "Times New Roman", "mathtext.sf": "Times New Roman",
            "mathtext.tt": "Times New Roman", "mathtext.fallback": None,
            "axes.unicode_minus": False,
        })
        self.audit["font_file"] = str(font)
        print(f"[font] {font}", flush=True)
        return font

    def read(self, relative):
        path = self.repo / relative
        raw = path.read_bytes()  # Missing files fail loudly, rather than producing an empty plot.
        self.audit["sources"][relative] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return json.loads(raw)

    def rows(self, relative, score, expected=422):
        data = self.read(relative)
        rows = data["scores"]
        if len(rows) != expected:
            raise ValueError(f"{relative}: expected {expected} rows, found {len(rows)}")
        def identity(row):
            return (row.get("source", ""), row.get("component", ""),
                    row.get("relative_path") or row.get("group") or row["path"])
        indexed = {identity(row): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate image identities: {relative}")
        keys = sorted(indexed)
        labels = np.array([indexed[k]["label"] for k in keys], dtype=int)
        values = np.array([indexed[k][score] for k in keys], dtype=float)
        if set(labels) != {0, 2} or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid labels or non-finite scores: {relative}")
        expected_counts = (259, 163) if expected == 422 else (24690, 12920)
        if ((labels == 0).sum(), (labels == 2).sum()) != expected_counts:
            raise ValueError(f"Unexpected class counts: {relative}")
        metrics = data["metrics"]
        candidate = metrics.get(score, metrics)
        npos, nneg = (labels == 2).sum(), (labels == 0).sum()
        auc = (rankdata(values)[labels == 2].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        if "auroc" in candidate and not np.isclose(auc, candidate["auroc"], atol=2e-5):
            raise ValueError(f"AUROC disagreement in {relative}/{score}: {auc} != {candidate['auroc']}")
        self.audit["checks"].append({"file": relative, "score": score,
                                     "images": len(rows), "auroc_recomputed": float(auc)})
        return data, keys, labels, values, [indexed[k] for k in keys]

    def save(self, fig, number):
        directory = self.out / f"figure{number}"
        directory.mkdir(parents=True, exist_ok=True)
        stem = directory / f"fig{number}_{STEMS[number]}"
        fig.savefig(stem.with_suffix(".png"), dpi=self.args.dpi, bbox_inches="tight", pad_inches=.035)
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.035)
        plt.close(fig)
        self.audit["outputs"].append(str(stem))
        print(f"[saved] {stem} (.png + .pdf)", flush=True)

    def csv(self, number, name, fields, rows):
        directory = self.out / f"figure{number}"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    @staticmethod
    def ecdf(ax, labels, values):
        for label, color, style in ((0, BLUE, "-"), (2, RED, "--")):
            x = np.sort(values[labels == label])
            y = np.arange(1, len(x) + 1) / len(x)
            # Keep every point: no KDE, outlier truncation, or score normalization.
            ax.step(np.r_[x[0], x], np.r_[0, y], where="post", color=color,
                    linestyle=style, label="Normal" if label == 0 else "Unknown")
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, .5, 1])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.grid(True, axis="y")

    def fig7(self):
        methods = [
            ("RT-DETR", "MSP score", "output/open_set_baselines/{c}_test_energy_msp_seed0.json", "msp_score"),
            ("PatchCore", "PatchCore score", "output/patchcore_wrn50_dim256_10pct_gpu_seed0/{c}_test.json", "patchcore_score"),
            ("Ours (DQCE)", "DQCE score", "output/final_results/dqce/{c}_seed0.json", "dqce_score"),
            ("Ours (QCS+semantic)", "Combined score", "output/{c}_test_qcs_semantic.json", "combined_score"),
            ("Ours (three-branch)", "Fused score", "output/{c}_three_branch_test.json", "three_branch_score")]
        fig, axes = plt.subplots(5, 3, figsize=(7.16, 8.0))
        reference = {}
        for row, (name, xlabel, pattern, score) in enumerate(methods):
            for col, condition in enumerate(CONDITIONS):
                _, keys, labels, values, _ = self.rows(pattern.format(c=condition), score)
                if condition in reference and (keys, labels.tolist()) != reference[condition]:
                    raise ValueError(f"Image/label mismatch: {name}, {condition}")
                reference[condition] = (keys, labels.tolist())
                ax = axes[row, col]
                self.ecdf(ax, labels, values)
                ax.set_xlabel(xlabel, labelpad=1)
                if row == 0:
                    ax.set_title(condition.title(), fontweight="bold", pad=7)
                if col == 0:
                    ax.set_ylabel("Cumulative fraction")
                    ax.text(0, 1.07, name, transform=ax.transAxes, fontsize=8, fontweight="bold")
                else:
                    ax.tick_params(labelleft=False)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2,
                   bbox_to_anchor=(.57, 1.005))
        fig.subplots_adjust(left=.085, right=.99, bottom=.055, top=.94, hspace=.83, wspace=.24)
        self.save(fig, 7)

    def fig8(self):
        data = self.read("output/final_results/dqce_ablation_3seed_summary.json")
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.9))
        records = []
        for panel, condition in enumerate(CONDITIONS):
            ax = axes.flat[panel]
            values = []
            for i, config in enumerate(CONFIGS):
                item = data[condition][config]["aupr"]
                seeds = np.asarray(item["seeds"])
                if not (np.isclose(np.mean(seeds), item["mean"]) and
                        np.isclose(np.std(seeds, ddof=1), item["std"])):
                    raise ValueError("Summary does not contain sample mean/SD")
                values.append(seeds * 100)
                ax.errorbar(i, item["mean"] * 100, yerr=item["std"] * 100,
                            fmt="_", ms=13, capsize=5, color=CONFIG_COLORS[i], lw=1.5, zorder=3)
                for seed, marker in enumerate(("o", "s", "^")):
                    ax.scatter(i + (seed - 1) * .10, seeds[seed] * 100, s=24,
                               marker=marker, color=CONFIG_COLORS[i], edgecolors="white", linewidth=.45, zorder=4)
                    records.append((condition, config, seed, seeds[seed],
                                    data[condition][config]["logit_std"]["seeds"][seed]))
            for seed in range(3):
                ax.plot(np.arange(3) + (seed - 1) * .10,
                        np.array(values)[:, seed], color=GRAY, alpha=.28, lw=.65, zorder=1)
            ax.set_xticks(range(3), ["Full", "w/o pair", "w/o known"])
            ax.set_xlim(-.4, 2.4)
            ax.set_ylabel("AUPR (%)")
            ax.set_title(f"({chr(97 + panel)}) {condition.title()}", loc="left", fontweight="bold")
            ax.grid(True, axis="y")
            ax.margins(y=.2)
        ax = axes[1, 1]
        for i, config in enumerate(CONFIGS):
            for j, condition in enumerate(CONDITIONS):
                item = data[condition][config]["logit_std"]
                ax.errorbar(j + (i - 1) * .19, item["mean"], yerr=item["std"],
                            fmt="_", ms=10, capsize=3, color=CONFIG_COLORS[i], lw=1)
                for seed, marker in enumerate(("o", "s", "^")):
                    ax.scatter(j + (i - 1) * .19 + (seed - 1) * .045,
                               item["seeds"][seed], s=15, color=CONFIG_COLORS[i], marker=marker,
                               edgecolors="white", linewidth=.35, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(range(3), [c.title() for c in CONDITIONS])
        ax.set_ylabel("Within-run logit SD")
        ax.set_title("(d) Logit dispersion", loc="left", fontweight="bold")
        ax.grid(True, axis="y", which="major")
        handles = [Line2D([], [], color=c, lw=2, label=l)
                   for c, l in zip(CONFIG_COLORS, CONFIG_LABELS)]
        seeds = [Line2D([], [], color=GRAY, marker=m, linestyle="", label=f"Seed {s}", ms=4)
                 for s, m in enumerate(("o", "s", "^"))]
        fig.legend(handles=handles + seeds, loc="lower center", ncol=3,
                   bbox_to_anchor=(.5, -.005), columnspacing=1.4)
        fig.subplots_adjust(left=.085, right=.99, top=.94, bottom=.20, hspace=.53, wspace=.28)
        self.csv(8, "ablation_points.csv", ["condition", "configuration", "seed", "aupr", "within_run_logit_sd"], records)
        self.save(fig, 8)

    def fig9(self):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5))
        for row, (protocol, prefix) in enumerate((("Mixed PLAMD OOD", "plamd_ood"),
                                                ("Snow-held-out OOD", "plamd_snow_ood"))):
            ref = None
            for col, (name, suffix) in enumerate((("RT-DETR", "baseline"), ("Ours (QCS–CSR)", "qcs_csr"))):
                _, keys, labels, values, _ = self.rows(f"output/{prefix}_{suffix}.json", "unknown_score", 37610)
                if ref is not None and ref != (keys, labels.tolist()):
                    raise ValueError(f"Protocol identity mismatch: {protocol}")
                ref = (keys, labels.tolist())
                ax = axes[row, col]
                self.ecdf(ax, labels, values)
                ax.set_xlabel("Archived image-level unknown score")
                ax.set_ylabel("Cumulative fraction")
                ax.set_title(f"({chr(97 + row * 2 + col)}) {name}", loc="left", fontweight="bold")
                ax.text(.02, .97, protocol, transform=ax.transAxes, va="top", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 2})
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2)
        fig.subplots_adjust(left=.08, right=.99, top=.94, bottom=.16, hspace=.48, wspace=.26)
        self.save(fig, 9)

    def fig10(self):
        # YOLO benchmark JSONs were not retained. These exact measurements were
        # supplied in the user's console records; do not replace them with model defaults.
        models = [
            ("YOLOv8m", .720, 5.774431254976662, 25.866742, BLUE, "o"),
            ("YOLO11m", .706, 6.548165284912102, 20.066886, TEAL, "s"),
            ("YOLOv12m", .691, 21.348839929996757, 21.723684, PURPLE, "D"),
            ("Faster R-CNN", .5845346105696102, 16.355072440055665, 41.439406, GRAY, "v"),
            ("RT-DETR", .7058676649555042, 14.190783314988948, 42.763962, GOLD, "^"),
            ("Ours", .7205192078770839, 15.926683805009816, 43.062464, RED, "*")]
        self.audit["detector_points_provenance"] = "User-supplied benchmark console logs and previously verified COCO evaluation caches; raw table values preserved."
        report = self.read("output/final_results/efficiency_report.json")
        fig = plt.figure(figsize=(7.16, 3.8))
        grid = fig.add_gridspec(2, 2, width_ratios=(1.6, 1), hspace=.65, wspace=.60)
        ax = fig.add_subplot(grid[:, 0])
        offsets = [(5, 10), (5, -17), (-8, -17), (0, 12), (-7, -20), (0, 11)]
        for (name, ap, latency, params, color, marker), offset in zip(models, offsets):
            ax.scatter(latency, ap * 100, s=params * 4, color=color, marker="o",
                       edgecolors=INK if name == "Ours" else "white", linewidth=.7, zorder=3)
            ax.annotate(name, (latency, ap * 100), xytext=offset, textcoords="offset points",
                        ha="right" if offset[0] < 0 else "left", fontsize=8,
                        fontweight="bold" if name == "Ours" else "normal")
        ax.set_xlim(3, 25)
        ax.set_ylim(56, 77)
        ax.set_xlabel("Detector-forward latency (ms/image)")
        ax.set_ylabel("Detection mAP50:95 (%)")
        ax.set_title("(a) Accuracy–latency tradeoff", loc="left", fontweight="bold")
        ax.grid(True)
        size_handles = [ax.scatter([], [], s=p * 4, facecolors="none", edgecolors=GRAY, label=f"{p} M") for p in (20, 40)]
        ax.legend(handles=size_handles, title="Marker area: parameters", loc="lower left",
                  fontsize=7, title_fontsize=8, handletextpad=1, labelspacing=1.1)
        for row, (field, xlabel, title) in enumerate((
                ("mean_seconds", "Full workflow time (s)", "(b) 422-image evaluation"),
                ("peak_gpu_memory_mb", "Peak GPU memory (MB)", "(c) Recorded memory run"))):
            a = fig.add_subplot(grid[row, 1])
            for j, (method, color) in enumerate((("DQCE", RED), ("PatchCore", BLUE))):
                value = report[method][field]
                a.hlines(j, 0, value, color=color, lw=5, alpha=.25)
                a.plot(value, j, "o", color=color, ms=5)
                if row == 0:
                    sd = report[method]["std_seconds"]
                    a.errorbar(value, j, xerr=sd, fmt="none", color=color, capsize=3)
                    text = f"{value:.2f} ± {sd:.2f}"
                else:
                    text = str(value)
                a.annotate(text, (value, j), xytext=(0, 9), textcoords="offset points",
                           ha="center", fontsize=8)
            a.set_yticks([0, 1], ["DQCE", "PatchCore"])
            a.set_ylim(-.5, 1.7)
            a.invert_yaxis()
            a.set_xlim(0, 110 if row == 0 else 3000)
            a.xaxis.set_major_locator(ticker.MaxNLocator(3))
            a.set_xlabel(xlabel)
            a.set_title(title, loc="left", fontweight="bold")
            a.grid(True, axis="x")
        fig.subplots_adjust(left=.08, right=.985, top=.92, bottom=.15)
        self.csv(10, "detector_cost_points.csv", ["model", "map50_95", "latency_ms", "parameters_million"],
                 [m[:4] for m in models])
        self.save(fig, 10)

    def fig11(self):
        from refine_fig11 import render_figure
        return render_figure(self)

    def fig11_without_insets(self):
        variants = [
            ("Ours (QCS+semantic, condition-calibrated)", "output/final_results/{c}_test_seed0.json", "combined_score", BLUE, "o", "-"),
            ("Ours (QCS+semantic, unified scoring)", "output/unified_{c}_test.json", "combined_score", TEAL, "s", "--"),
            ("Ours (three-branch)", "output/{c}_three_branch_test.json", "three_branch_score", RED, "D", "-.")]
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.7))
        records = []
        for col, condition in enumerate(CONDITIONS):
            ref = None
            for name, pattern, field, color, marker, style in variants:
                data, keys, labels, scores, _ = self.rows(pattern.format(c=condition), field)
                if ref is not None and ref != (keys, labels.tolist()):
                    raise ValueError(f"Rejection identity mismatch: {condition}")
                ref = (keys, labels.tolist())
                unknown = labels == 2
                order = np.argsort(scores, kind="stable")
                sorted_scores = scores[order]
                # Accept entire equal-score groups, never separate tied observations.
                endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0) + 1, len(scores)]
                accepted_unknown = np.cumsum(unknown[order])[endpoints - 1]
                coverage = endpoints / len(scores)
                risk = accepted_unknown / endpoints
                recall = 1 - accepted_unknown / unknown.sum()
                saved = data["metrics"]["conformal"]
                threshold = saved["threshold"]
                accepted = scores < threshold
                point = [accepted.mean(), unknown[accepted].mean(), (~accepted & unknown).sum() / unknown.sum()]
                expected = [saved[k] for k in ("coverage", "selective_risk", "unknown_recall")]
                if not np.allclose(point, expected, atol=1e-6):
                    raise ValueError(f"Stored rejection point mismatch: {condition}/{name}: {point} != {expected}")
                for row, y in enumerate((risk, recall)):
                    ax = axes[row, col]
                    ax.step(coverage, y, where="post", color=color, linestyle=style, label=name)
                    ax.scatter(point[0], point[row + 1], marker=marker, color=color,
                               edgecolors="white", linewidth=.7, s=32, zorder=5)
                records.append([condition, name, threshold, *point])
            for row in range(2):
                ax = axes[row, col]
                ax.set_xlim(0, 1.01)
                ax.set_ylim(-.025, 1.025)
                ax.set_xticks([0, .5, 1])
                ax.set_yticks([0, .5, 1])
                ax.grid(True)
                ax.set_xlabel("Acceptance coverage")
                if col == 0:
                    ax.set_ylabel("Selective risk" if row == 0 else "Unknown recall")
                else:
                    ax.tick_params(labelleft=False)
                ax.set_title(f"({chr(97 + row * 3 + col)}) {condition.title()}", loc="left", fontweight="bold")
            axes[0, col].axhline(163 / 422, color=GRAY, ls=":", lw=.8)
        handles = [Line2D([], [], color=v[3], ls=v[5], marker=v[4], ms=4, label=v[0]) for v in variants]
        fig.legend(handles=handles, loc="lower center", ncol=1, bbox_to_anchor=(.55, -.005),
                   handlelength=2.5, labelspacing=.25)
        fig.subplots_adjust(left=.08, right=.99, top=.94, bottom=.25, hspace=.55, wspace=.22)
        self.csv(11, "calibrated_operating_points.csv", ["condition", "configuration", "threshold", "coverage", "selective_risk", "unknown_recall"], records)
        self.audit["checks"].append("Fig11: all 9 archived operating points reproduced with accepted = score < threshold; tied scores kept together.")
        self.save(fig, 11)

    def fig12(self):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.05))
        weights = []
        matrices = []
        for condition in CONDITIONS:
            data, _, _, _, rows = self.rows(f"output/{condition}_three_branch_test.json", "three_branch_score")
            w = np.array([data["metrics"]["weights"][b] for b in ("qcs", "semantic", "dqce")])
            if not np.isclose(w.sum(), 1):
                raise ValueError("Fusion coefficients do not sum to one")
            branches = np.array([[r[k] for k in ("qcs_score", "semantic_score", "dqce_score")] for r in rows])
            mu = np.array(data["metrics"]["calibration_stats"]["mean"])
            sd = np.array(data["metrics"]["calibration_stats"]["std"])
            fused = ((branches - mu) / sd) @ w
            if not np.allclose(fused, [r["three_branch_score"] for r in rows], atol=1e-5):
                raise ValueError("Stored fusion scores cannot be reproduced from saved branch scores and calibration stats")
            matrix = spearmanr(branches, axis=0).statistic
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Undefined branch correlation in {condition}")
            weights.append(w)
            matrices.append(matrix)
            self.csv(12, f"{condition}_spearman.csv", ["branch", "QCS", "Semantic", "DQCE"],
                     [[b, *matrix[i]] for i, b in enumerate(("QCS", "Semantic", "DQCE"))])
        ax = axes[0, 0]
        bottom = np.zeros(3)
        for i, (branch, color, hatch) in enumerate(zip(("QCS", "Semantic", "DQCE"), (BLUE, TEAL, RED), ("", "//", ".."))):
            heights = np.array(weights)[:, i]
            ax.bar(range(3), heights, bottom=bottom, width=.6, color=color, hatch=hatch,
                   edgecolor="white", linewidth=.6, label=branch)
            for j, height in enumerate(heights):
                if height >= .15:
                    ax.text(j, bottom[j] + height / 2, f"{height:.2f}", ha="center", va="center", color="white", fontsize=9)
                elif height > 0:
                    ax.annotate(f"{height:.2f}", (j + .30, bottom[j] + height / 2),
                                xytext=(j + .48, .08 if i == 0 else .27),
                                va="center", fontsize=8,
                                arrowprops={"arrowstyle": "-", "color": INK, "lw": .6})
                else:
                    ax.annotate("DQCE = 0", (j, 1), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7)
            bottom += heights
        ax.set_xticks(range(3), [c.title() for c in CONDITIONS])
        ax.set_xlim(-.55, 2.95)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, .5, 1])
        ax.set_ylabel("Fusion coefficient")
        ax.set_title("(a) Condition-specific weights", loc="left", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3, columnspacing=.8)
        for i, (condition, matrix) in enumerate(zip(CONDITIONS, matrices)):
            ax = axes.flat[i + 1]
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            for y in range(3):
                for x in range(3):
                    ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center",
                            color="white" if abs(matrix[y, x]) > .55 else INK, fontsize=10)
            ax.set_xticks(range(3), ["QCS", "Semantic", "DQCE"])
            ax.set_yticks(range(3), ["QCS", "Semantic", "DQCE"])
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-.5, 3), minor=True)
            ax.set_yticks(np.arange(-.5, 3), minor=True)
            ax.grid(which="minor", color="white", lw=2)
            ax.tick_params(which="minor", length=0)
            ax.set_title(f"({chr(98 + i)}) {condition.title()}", loc="left", fontweight="bold")
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=.08, right=.9, top=.93, bottom=.08, hspace=.58, wspace=.34)
        cax = fig.add_axes([.935, .19, .018, .59])
        cb = fig.colorbar(im, cax=cax, ticks=[-1, -.5, 0, .5, 1])
        cb.set_label("Spearman correlation", labelpad=6)
        self.csv(12, "fusion_weights.csv", ["condition", "QCS", "Semantic", "DQCE"],
                 [[c, *w] for c, w in zip(CONDITIONS, weights)])
        self.save(fig, 12)

    def fig6(self):
        if self.args.fig6_layout == "detail":
            from refine_fig6 import render_detail
            return render_detail(self)
        base = "output/final_results/figures"
        manifest = self.read(f"{base}/detection_image_manifest.json")
        if len(manifest) != 4:
            raise ValueError("Fig6 expects the four selected validation cases")
        paths = [("YOLOv8m", "yolo/yolov8m"), ("YOLO11m", "yolo/yolo11m"),
                 ("YOLOv12m", "yolo/yolo12m"), ("Faster R-CNN", "other/faster_rcnn"),
                 ("RT-DETR", "other/rtdetr_baseline"), ("Ours", "other/rtdetr_ours")]
        predictions = {name: {r["id"]: r for r in self.read(f"{base}/{folder}/predictions.json")}
                       for name, folder in paths}
        annotations = self.read(str(self.args.annotation))
        categories = {int(c["id"]): c["name"] for c in annotations["categories"]}
        canon = lambda s: " ".join(s.lower().replace("-", " ").split())
        category_by_name = {canon(name): cid for cid, name in categories.items()}
        records = []
        images, gt, crops = [], [], []
        for case in manifest:
            imagepath = Path(case["image"])
            if not imagepath.exists():
                imagepath = self.args.image_root / imagepath.name
            image = Image.open(imagepath).convert("RGB")
            matches = [v for v in annotations["images"] if Path(v["file_name"]).name == imagepath.name]
            if len(matches) != 1:
                raise ValueError(f"Expected unique GT match: {imagepath}")
            annotation = [v for v in annotations["annotations"] if v["image_id"] == matches[0]["id"] and not v.get("iscrowd", 0)]
            detections = []
            for v in annotation:
                x, y, w, h = v["bbox"]
                detections.append({"bbox": [x, y, x + w, y + h], "cid": v["category_id"], "score": None})
            if not detections:
                raise ValueError(f"No ground truth: {imagepath}")
            target = min(detections, key=lambda v: (v["bbox"][2] - v["bbox"][0]) * (v["bbox"][3] - v["bbox"][1]))["bbox"]
            x, y, x2, y2 = target
            w = max((x2 - x) * 2.0, image.width * .16)
            h = max((y2 - y) * 2.0, image.height * .20)
            # Same rectangular zoom for every method; deterministic smallest-GT target.
            w, h = min(w, image.width * .72), min(h, image.height * .72)
            left = max(0, min((x + x2 - w) / 2, image.width - w))
            top = max(0, min((y + y2 - h) / 2, image.height - h))
            crops.append(tuple(map(int, (left, top, left + w, top + h))))
            images.append(image)
            gt.append(detections)
            records.append({"id": case["id"], "image": str(imagepath), "zoom_xyxy": crops[-1]})
        used = set(d["cid"] for case in gt for d in case)
        all_dets = {"Ground truth": gt}
        for name, _ in paths:
            all_dets[name] = []
            for case in manifest:
                saved = predictions[name][case["id"]]
                if Path(saved["image"]).name != Path(case["image"]).name:
                    raise ValueError("Prediction image identity mismatch")
                dets = []
                for d in saved["detections"]:
                    if d["score"] < self.args.conf:
                        continue
                    # Class names, rather than incompatible 0/1-based numeric labels,
                    # establish the common color mapping across detector families.
                    cid = category_by_name[canon(d["class_name"])]
                    used.add(cid)
                    dets.append({"bbox": d["bbox"], "score": d["score"], "cid": cid})
                all_dets[name].append(dets)
        colors = [BLUE, RED, TEAL, GOLD, PURPLE, "#56B4E9", "#6B5B95", "#8C564B", "#407C70", "#A96B20", "#5376B9", "#A45470", "#515C66", "#925A36", "#596A2E", "#4C8899", "#7B63A5", "#343C49"]
        color = {cid: colors[i % len(colors)] for i, cid in enumerate(sorted(categories))}
        names = ["Ground truth"] + [p[0] for p in paths]
        fig, axes = plt.subplots(7, 4, figsize=(7.16, 8.05))
        standalone = self.out / "figure6" / "individual"
        standalone.mkdir(parents=True, exist_ok=True)
        for row, name in enumerate(names):
            for col, image in enumerate(images):
                ax = axes[row, col]
                # Rasterize only at the resolution needed for a 1.6-inch panel;
                # retain original coordinates for every box and zoom window.
                thumb = image.resize((820, 461), Image.Resampling.LANCZOS)
                ax.imshow(thumb, extent=(0, image.width, image.height, 0))
                ax.set_xlim(0, image.width)
                ax.set_ylim(image.height, 0)
                ax.set_axis_off()
                dets = all_dets[name][col]
                for d in dets:
                    x, y, x2, y2 = d["bbox"]
                    ax.add_patch(Rectangle((x, y), x2 - x, y2 - y, fill=False, ec=color[d["cid"]], lw=.8))
                    # Color-only boxes keep the dense montage readable. Full class
                    # names and confidence scores are in the standalone exports.
                left, top, right, bottom = crops[col]
                ax.add_patch(Rectangle((left, top), right-left, bottom-top, fill=False,
                                       ec="white", ls="--", lw=.65))
                inset = ax.inset_axes([.63, .03, .35, .36])
                zoom = image.crop(crops[col])
                zoom.thumbnail((400, 250), Image.Resampling.LANCZOS)
                inset.imshow(zoom, extent=(left, right, bottom, top), aspect="equal")
                for d in dets:
                    x, y, x2, y2 = d["bbox"]
                    inset.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, ec=color[d["cid"]], lw=.8))
                inset.set_xlim(left, right)
                inset.set_ylim(bottom, top)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_visible(True)
                    spine.set_color("white")
                    spine.set_linewidth(.7)
                if row == 0:
                    ax.set_title(["(a) Easy", "(b) Crowded", "(c) Small object", "(d) Overlap"][col], fontsize=9, fontweight="bold", pad=4)
                if col == 0:
                    ax.text(-.03, .5, name, transform=ax.transAxes, rotation=90,
                            ha="right", va="center", fontsize=8, fontweight="bold" if name == "Ours" else "normal")
                # Full-resolution standalone files retain full class names + scores.
                full = image.copy()
                painter = ImageDraw.Draw(full)
                font = ImageFont.truetype(str(self.font), 30)
                for d in dets:
                    x, y, x2, y2 = d["bbox"]
                    painter.rectangle((x, y, x2, y2), outline=color[d["cid"]], width=5)
                    title = categories[d["cid"]] + (f" {d['score']:.2f}" if d["score"] is not None else "")
                    box = painter.textbbox((0, 0), title, font=font)
                    tw, th = box[2] + 8, box[3] - box[1] + 8
                    tx, ty = min(max(x, 0), max(0, image.width-tw)), max(0, y-th)
                    painter.rectangle((tx, ty, tx+tw, ty+th), fill=color[d["cid"]])
                    painter.text((tx+4, ty+4-box[1]), title, font=font, fill="white")
                sub = standalone / name.replace(" ", "_")
                sub.mkdir(exist_ok=True)
                full.save(sub / f"{manifest[col]['id']}.png", dpi=(self.args.dpi, self.args.dpi))
        handles = [Line2D([], [], color=color[cid], lw=2, label=categories[cid]) for cid in sorted(used)]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
                   handlelength=1.0, columnspacing=.9, labelspacing=.5, bbox_to_anchor=(.5, -.003))
        fig.subplots_adjust(left=.055, right=.995, top=.97, bottom=.075, hspace=.045, wspace=.025)
        self.audit["detection_cases"] = records
        self.audit["detection_confidence"] = self.args.conf
        self.audit["detection_color_mapping"] = {f"C{k:02} {v}": color[k] for k, v in categories.items()}
        self.save(fig, 6)

    def run(self):
        self.out.mkdir(parents=True, exist_ok=True)
        failed = []
        for number in self.args.figures:
            try:
                getattr(self, f"fig{number}")()
            except Exception as exc:
                failed.append((number, f"{type(exc).__name__}: {exc}"))
                print(f"[FAILED] Fig.{number}: {exc}", flush=True)
                traceback.print_exc()
                plt.close("all")
        self.audit["failures"] = failed
        (self.out / "generation_audit.json").write_text(json.dumps(self.audit, indent=2, ensure_ascii=False), encoding="utf-8")
        if failed:
            raise SystemExit(f"No blank figures were created. Resolve the {len(failed)} reported data error(s) and rerun.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True, help="Local RT-DETRv2 project/output root")
    p.add_argument("--figure-root", type=Path, required=True, help="Root containing figure6/ ... figure12/")
    p.add_argument("--font-dir", type=Path)
    p.add_argument("--figures", type=int, nargs="+", choices=range(6, 13), default=list(range(6, 13)))
    p.add_argument("--annotation", type=Path, default=Path("data/InsPLAD/annotations/instances_val_insplad17.json"))
    p.add_argument("--image-root", type=Path, default=Path("data/InsPLAD/val"))
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--fig6-layout", choices=("overview", "detail"), default="detail")
    p.add_argument("--dpi", type=int, default=450)
    args = p.parse_args()
    if args.dpi < 300 or not 0 <= args.conf <= 1:
        p.error("Use dpi >= 300 and a confidence threshold in [0, 1].")
    Artwork(args).run()


if __name__ == "__main__":
    main()
