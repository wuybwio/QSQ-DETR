"""Detail renderer for Fig.6. Uses cached predictions without modifying them.

Targets are chosen by the largest inter-model spread of same-class box IoU,
independent of which model wins. TP/FP/FN are full-image, score-ordered,
class-aware greedy diagnostic matches at IoU >= 0.50; they are not COCO AP.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


def box_iou(box, other):
    box, other = np.asarray(box, float), np.asarray(other, float)
    intersection = np.prod(np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])))
    area = np.prod(np.maximum(0, box[2:] - box[:2]))
    other_area = np.prod(np.maximum(0, other[2:] - other[:2]))
    return float(intersection / (area + other_area - intersection)) if area + other_area > intersection else 0.0


def match(gt, predictions, threshold=.5):
    available = set(range(len(gt)))
    assignments = {}
    for pi in sorted(range(len(predictions)), key=lambda i: -predictions[i]["score"]):
        pred = predictions[pi]
        possible = [(box_iou(pred["bbox"], gt[gi]["bbox"]), gi) for gi in available
                    if gt[gi]["cid"] == pred["cid"]]
        if possible:
            overlap, gi = max(possible)
            if overlap >= threshold:
                available.remove(gi)
                assignments[gi] = (pi, overlap)
    counts = (len(assignments), len(predictions) - len(assignments), len(available))
    assert counts[0] + counts[2] == len(gt)
    assert counts[0] + counts[1] == len(predictions)
    return assignments, counts


def make_crop(box, size):
    width, height = size
    x, y, x2, y2 = box
    ratio = width / height
    crop_w = min(width, max(320, (x2 - x) * 1.9, (y2 - y) * 1.9 * ratio))
    crop_h = crop_w / ratio
    left = min(max(0, (x + x2 - crop_w) / 2), width - crop_w)
    top = min(max(0, (y + y2 - crop_h) / 2), height - crop_h)
    return (left, top, left + crop_w, top + crop_h)


def touches(box, crop):
    return box[0] < crop[2] and box[2] > crop[0] and box[1] < crop[3] and box[3] > crop[1]


def render_detail(art):
    from generate_experiment_figures import BLUE, RED, TEAL, GOLD, PURPLE, INK
    base = "output/final_results/figures"
    manifest = art.read(f"{base}/detection_image_manifest.json")
    sources = [("YOLOv8m", "yolo/yolov8m"), ("YOLO11m", "yolo/yolo11m"),
               ("YOLOv12m", "yolo/yolo12m"), ("Faster R-CNN", "other/faster_rcnn"),
               ("RT-DETR", "other/rtdetr_baseline"), ("Ours", "other/rtdetr_ours")]
    saved = {}
    for name, folder in sources:
        rows = art.read(f"{base}/{folder}/predictions.json")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate prediction case IDs: {name}")
        saved[name] = {r["id"]: r for r in rows}
    annotations = art.read(str(art.args.annotation))
    cats = {c["id"]: c["name"] for c in annotations["categories"]}
    canon = lambda value: " ".join(value.lower().replace("-", " ").split())
    cid_by_name = {canon(v): k for k, v in cats.items()}
    palette = [BLUE, RED, TEAL, GOLD, PURPLE, "#56B4E9", "#6B5B95", "#8C564B",
               "#407C70", "#A96B20", "#5376B9", "#A45470", "#515C66", "#925A36",
               "#596A2E", "#4C8899", "#7B63A5", "#343C49"]
    colors = {cid: palette[i % len(palette)] for i, cid in enumerate(sorted(cats))}
    names = ["Ground truth"] + [name for name, _ in sources]
    cases, metrics_rows, selection_rows, target_rows = [], [], [], []
    used = set()
    for case in manifest:
        path = Path(case["image"])
        if not path.exists():
            path = art.args.image_root / path.name
        image = Image.open(path).convert("RGB")
        image_info = [r for r in annotations["images"] if Path(r["file_name"]).name == path.name]
        if len(image_info) != 1:
            raise ValueError(f"Image not uniquely matched to GT: {path}")
        gt = []
        for row in annotations["annotations"]:
            if row["image_id"] != image_info[0]["id"] or row.get("iscrowd", 0) or row.get("ignore", 0):
                continue
            x, y, w, h = row["bbox"]
            gt.append({"bbox": [x, y, x+w, y+h], "cid": row["category_id"], "ann_id": row["id"]})
        gt.sort(key=lambda g: g["ann_id"])
        predictions, matches, counts = {}, {}, {}
        quality = []
        for name, _ in sources:
            row = saved[name][case["id"]]
            if Path(row["image"]).name != path.name:
                raise ValueError("Prediction image identity mismatch")
            predictions[name] = [dict(bbox=list(map(float, d["bbox"])),
                                      cid=cid_by_name[canon(d["class_name"])], score=float(d["score"]))
                                 for d in row["detections"] if d["score"] >= art.args.conf]
            for d in predictions[name]:
                if not np.all(np.isfinite(d["bbox"])) or d["bbox"][2] <= d["bbox"][0] or d["bbox"][3] <= d["bbox"][1]:
                    raise ValueError("Invalid prediction box")
            matches[name], counts[name] = match(gt, predictions[name])
            metrics_rows.append([case["id"], name, len(gt), len(predictions[name]), *counts[name]])
            quality.append([max([box_iou(g["bbox"], d["bbox"]) for d in predictions[name]
                                 if g["cid"] == d["cid"]] or [0.0]) for g in gt])
        quality = np.asarray(quality)
        spread = np.ptp(quality, axis=0)
        target = int(np.argmax(spread))
        crop = make_crop(gt[target]["bbox"], image.size)
        selection_rows.append({"case": case["id"], "annotation_id": gt[target]["ann_id"],
                               "category": cats[gt[target]["cid"]], "roi_xyxy": crop,
                               "max_minus_min_best_class_iou": float(spread[target]),
                               "best_class_iou_by_model": dict(zip(names[1:], quality[:, target].tolist()))})
        for name, _ in sources:
            assignment = matches[name].get(target)
            target_rows.append([case["id"], gt[target]["ann_id"], cats[gt[target]["cid"]], name,
                                "TP" if assignment else "FN", assignment[1] if assignment else "",
                                predictions[name][assignment[0]]["score"] if assignment else ""])
        used.update(g["cid"] for g in gt if touches(g["bbox"], crop))
        for detections in predictions.values():
            used.update(d["cid"] for d in detections if touches(d["bbox"], crop))
        cases.append((case, image, gt, predictions, matches, counts, target, crop))
    if len(cases) != 4:
        raise ValueError("Detail layout expects four cases")
    fig = plt.figure(figsize=(7.16, 8.35))
    grid = fig.add_gridspec(7, 4, left=.058, right=.995, top=.953, bottom=.105, hspace=.40, wspace=.045)
    export_root = art.out / "figure6" / "individual_detail"
    export_root.mkdir(parents=True, exist_ok=True)
    context_fig, context_axes = plt.subplots(1, 4, figsize=(7.16, 1.35))
    for col, (case, image, gt, predictions, matches, counts, target, crop) in enumerate(cases):
        left, top, right, bottom = crop
        crop_image = image.crop(tuple(map(int, crop)))
        crop_image.thumbnail((850, 480), Image.Resampling.LANCZOS)
        target_box = gt[target]["bbox"]
        ax = context_axes[col]
        ax.imshow(image.resize((820, 461)))
        x, y, x2, y2 = np.array(crop) * np.array([820/image.width, 461/image.height]*2)
        ax.add_patch(Rectangle((x, y), x2-x, y2-y, ec=GOLD, fill=False, lw=1.2))
        ax.set_title(["(a) Easy", "(b) Crowded", "(c) Small object", "(d) Overlap"][col], fontsize=9)
        ax.set_axis_off()
        for row, name in enumerate(names):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(crop_image, extent=(left, right, bottom, top))
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
            ax.set_axis_off()
            detections = gt if row == 0 else predictions[name]
            for d in detections:
                if not touches(d["bbox"], crop):
                    continue
                x, y, x2, y2 = d["bbox"]
                ax.add_patch(Rectangle((x, y), x2-x, y2-y, fill=False, ec=colors[d["cid"]], lw=1.0))
            if row > 0:
                x, y, x2, y2 = target_box
                reference = Rectangle((x, y), x2-x, y2-y, fill=False, ec="white", lw=.9, ls=(0, (3, 2)))
                reference.set_path_effects([pe.Stroke(linewidth=2.0, foreground=INK), pe.Normal()])
                ax.add_patch(reference)
                assignment = matches[name].get(target)
                if assignment:
                    pi, overlap = assignment
                    detail = f"IoU {overlap:.2f} | p {predictions[name][pi]['score']:.2f}"
                    text_color = INK
                else:
                    detail = "Target missed (FN)"
                    text_color = RED
                tp, fp, fn = counts[name]
                ax.text(.5, -.03, f"TP {tp}   FP {fp}   FN {fn}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7, color=INK)
                ax.text(.5, -.19, detail, transform=ax.transAxes, ha="center", va="top",
                        fontsize=7, color=text_color, fontweight="bold" if assignment is None else "normal")
            else:
                # Original scene is available as a separate context strip, so no
                # overlay can obscure a relevant target inside this reference row.
                ax.text(.5, -.05, f"Target: {cats[gt[target]['cid']]}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7, wrap=True)
            if row == 0:
                ax.set_title(["(a) Easy", "(b) Crowded", "(c) Small object", "(d) Overlap"][col],
                             fontsize=9, fontweight="bold", pad=4)
            if col == 0:
                ax.text(-.04, .5, name, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=8, fontweight="bold" if name == "Ours" else "normal")
            # Export each exact panel (including diagnostic annotation) for LaTeX assembly.
            directory = export_root / name.replace(" ", "_")
            directory.mkdir(exist_ok=True)
            # Independent panel figures avoid rasterizing the whole montage 28 times.
            panel_fig, pa = plt.subplots(figsize=(3.5, 2.45))
            pa.imshow(crop_image, extent=(left, right, bottom, top))
            for patch in ax.patches:
                copied = Rectangle(patch.get_xy(), patch.get_width(), patch.get_height(),
                                   fill=False, ec=patch.get_edgecolor(), lw=patch.get_linewidth()*1.7,
                                   ls=patch.get_linestyle())
                if patch.get_path_effects():
                    copied.set_path_effects(patch.get_path_effects())
                pa.add_patch(copied)
            pa.set_xlim(left, right)
            pa.set_ylim(bottom, top)
            pa.set_axis_off()
            pa.set_title(name, fontsize=11, fontweight="bold")
            if row > 0:
                pa.set_title(f"{name}\nTP {tp}  FP {fp}  FN {fn} | {detail}", fontsize=10)
            panel_fig.subplots_adjust(left=.01, right=.99, top=.80, bottom=.01)
            panel_fig.savefig(directory / f"{case['id']}.png", dpi=art.args.dpi, bbox_inches="tight", pad_inches=.03)
            plt.close(panel_fig)
    handles = [Line2D([], [], color=colors[cid], lw=2, label=cats[cid]) for cid in sorted(used)]
    handles.append(Line2D([], [], color=INK, ls="--", lw=1, label="Target GT reference"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               handlelength=1.2, columnspacing=.9, labelspacing=.4, bbox_to_anchor=(.52, .02))
    fig.text(.53, .002, f"Counts: full-image diagnostic matches at IoU ≥ 0.50; p: confidence; identical crops and confidence ≥ {art.args.conf:.2f}.",
             ha="center", fontsize=7)
    context_fig.subplots_adjust(left=.01, right=.99, top=.82, bottom=.04, wspace=.03)
    context_fig.savefig(art.out / "figure6" / "fig6_context.png", dpi=art.args.dpi, bbox_inches="tight", pad_inches=.03)
    plt.close(context_fig)
    art.csv(6, "detection_diagnostics.csv", ["case", "model", "gt_count", "prediction_count", "TP", "FP", "FN"], metrics_rows)
    art.csv(6, "target_localization.csv", ["case", "target_annotation", "target_category", "model", "status", "matched_iou", "confidence"], target_rows)
    art.audit["fig6_detail"] = {
        "checkpoint_provenance": "Cached predictions; the source JSONs do not record checkpoint identity. No fresh inference performed.",
        "target_selection": "Largest max-minus-min best same-class IoU across all six models, independent of winner; annotation ID breaks ties.",
        "matching": "Full-image, score-ordered, same-class greedy matching at IoU >= 0.50. Not a replacement for COCO AP.",
        "cases": selection_rows,
        "notes": "Predicted boxes, classes, confidences and display threshold unchanged; one shared ROI per image."}
    (art.out / "figure6" / "detail_audit.json").write_text(json.dumps(art.audit, ensure_ascii=False, indent=2), encoding="utf-8")
    art.save(fig, 6)
