"""Redraw the two manuscript figures from verified, unchanged study results.

Run independently of the legacy table generator. PDF and SVG contain vector
marks and text; 600-dpi PNGs are exported directly for figure-only inspection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

BASE = Path(__file__).resolve().parents[1]
MANUSCRIPT = BASE / "generated"
RESULTS = BASE / "04_分析结果/正式研究"
MODELS = ["Qwen3-4B-Instruct-2507-4bit", "Phi-4-mini-instruct-4bit"]
NAMES = {MODELS[0]: "Qwen3-4B", MODELS[1]: "Phi-4-mini"}
METHODS = ["direct", "structured"]
LABELS = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]
DISPLAY_LABELS = ["Supported", "Contradicted", "Insufficient"]
ANALYSES = [
    ("all_heldout", "All held-out claims", "36 claims"),
    ("shared_draft_and_commencement_equal_group_weight", "Shared-question weighting", "29 weighted units"),
    ("exclude_shared_amendment_or_draft_inputs", "Exclude shared inputs", "18 claims"),
    ("exclude_four_outcome_wording_cases", "Exclude outcome claims", "32 claims"),
]
INK = "#243A43"
MUTED = "#53666E"
GRID = "#DAE3E6"
TEAL = "#176B79"
COPPER = "#A35F2F"
COLORS = {MODELS[0]: TEAL, MODELS[1]: COPPER}
CMAP = LinearSegmentedColormap.from_list(
    "marine_share", ["#F3F7F8", "#C7E0E4", "#74B0BB", "#287D8C", "#07505F"]
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data():
    result_path = RESULTS / "正式研究结果.json"
    rows_path = RESULTS / "正式逐条分析.json"
    case_path = BASE / "04_分析结果/法规命题基准/cases_v1.json"
    result = json.loads(result_path.read_text())
    rows = json.loads(rows_path.read_text())
    cases = {c["case_id"]: c for c in json.loads(case_path.read_text())}
    held = [r for r in rows if r["split"] == "heldout"]
    assert len(held) == 144
    assert len({(r["model"], r["method"], r["case_id"]) for r in held}) == 144
    for r in held:
        assert r["gold"] == cases[r["case_id"]]["gold_label"]
        assert r["correct"] == (r["gold"] == r["prediction"])
        assert r["shared_instrument"] == any(
            x.split(":")[0] in {"AMD", "DRF"} for x in cases[r["case_id"]]["evidence_ids"]
        )
    summaries = {(g["model"], g["method"]): g for g in result["splits"]["heldout"]["groups"]}
    panels = []
    for model in MODELS:
        for method in METHODS:
            selected = [r for r in held if r["model"] == model and r["method"] == method]
            matrix = [[sum(r["gold"] == g and r["prediction"] == p for r in selected)
                       for p in LABELS] for g in LABELS]
            archived = summaries[model, method]
            assert matrix == [[archived["confusion_matrix"][g][p] for p in LABELS] for g in LABELS]
            assert [sum(row) for row in matrix] == [18, 9, 9]
            correct = sum(matrix[i][i] for i in range(3))
            assert correct == archived["correct"]
            shares = [[100 * value / sum(row) for value in row] for row in matrix]
            panels.append(dict(model=model, method=method, counts=matrix,
                               row_percentages=shares, correct=correct, accuracy=100 * correct / 36))

    families = Counter(c["family"] for c in cases.values() if c["split"] == "heldout")
    weights = {c["case_id"]: Fraction(1, families[c["family"]])
               if c["family"] in {"draft_status", "effective_date"} else Fraction(1)
               for c in cases.values() if c["split"] == "heldout"}
    assert sum(weights.values()) == 29

    def difference(model, predicate, weighted=False):
        scores = {}
        for method in METHODS:
            selected = [r for r in held if r["model"] == model and r["method"] == method and predicate(r)]
            denominator = sum((weights[r["case_id"]] if weighted else Fraction(1)) for r in selected)
            numerator = sum((weights[r["case_id"]] if weighted else Fraction(1))
                            for r in selected if r["correct"])
            scores[method] = numerator / denominator
        return float(100 * (scores["structured"] - scores["direct"]))

    sensitivity = []
    leave_one_out = []
    for model in MODELS:
        for analysis, _, _ in ANALYSES:
            predicate = lambda r: True
            if analysis == "exclude_shared_amendment_or_draft_inputs":
                predicate = lambda r: not r["shared_instrument"]
            elif analysis == "exclude_four_outcome_wording_cases":
                predicate = lambda r: r["family"] != "law_to_outcome_gap"
            value = difference(model, predicate, analysis == "shared_draft_and_commencement_equal_group_weight")
            archived = next(v for v in result["sensitivity"] if v["model"] == model and v["analysis"] == analysis)
            assert abs(value - 100 * archived["accuracy_difference"]) < 1e-10
            sensitivity.append(dict(model=model, analysis=analysis, percentage_points=value))
        for mpa in sorted({r["mpa_id"] for r in held}):
            value = difference(model, lambda r, mpa=mpa: r["mpa_id"] != mpa)
            archived = next(v for v in result["leave_one_mpa_out"]
                            if v["model"] == model and v["excluded_mpa"] == mpa)
            assert abs(value - 100 * archived["accuracy_difference"]) < 1e-10
            leave_one_out.append(dict(model=model, excluded_mpa=mpa, percentage_points=value))
    return dict(panels=panels, sensitivity=sensitivity, leave_one_mpa_out=leave_one_out,
                source_sha256={str(p.relative_to(BASE)): digest(p) for p in [result_path, rows_path, case_path]},
                normalization="Counts and percentages within each reference class",
                interval_definition="Full range over nine MPA exclusions, not a confidence interval")


def luminance(rgb):
    return sum(w * (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
               for w, v in zip([0.2126, 0.7152, 0.0722], rgb))


def contrast_text(face):
    light = luminance(face[:3])
    dark = luminance(to_rgb("black"))
    white_ratio = 1.05 / (light + 0.05)
    dark_ratio = (light + 0.05) / (dark + 0.05)
    return ("white", white_ratio) if white_ratio >= dark_ratio else ("black", dark_ratio)


def canvas_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            outside.append(artist.get_text())
    assert not outside, f"Text outside figure canvas: {outside}"
    return dict(canvas_inches=list(fig.get_size_inches()), text_outside_canvas=outside)


def save_figure(fig, stem, output):
    check = canvas_check(fig)
    metadata = {"Title": "Classification outcomes" if stem == "heldout_confusion" else "Sensitivity to case selection",
                "Author": "", "Creator": "Matplotlib", "CreationDate": None, "ModDate": None}
    fig.savefig(output / f"{stem}.pdf", metadata=metadata)
    fig.savefig(output / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(output / f"{stem}.png", dpi=600)
    plt.close(fig)
    return check


def confusion_figure(data, output):
    width, height = 7.2, 5.85
    fig = plt.figure(figsize=(width, height))
    side = 2.10
    lefts = [1.08, 4.39]
    bottoms = [3.24, 0.49]
    contrasts = []
    for index, panel in enumerate(data["panels"]):
        row, col = divmod(index, 2)
        x, y = lefts[col], bottoms[row]
        ax = fig.add_axes([x / width, y / height, side / width, side / height])
        ax.set(xlim=(-0.5, 2.5), ylim=(2.5, -0.5), aspect="equal")
        for i in range(3):
            for j in range(3):
                count = panel["counts"][i][j]
                pct = panel["row_percentages"][i][j]
                face = CMAP(pct / 100)
                text_color, contrast = contrast_text(face)
                contrasts.append(contrast)
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=face, edgecolor="white", linewidth=1.5))
                if i == j:
                    ax.add_patch(Rectangle((j - .455, i - .455), .91, .91, fill=False,
                                           edgecolor=text_color, linewidth=.8, alpha=.72))
                ax.text(j, i - .10, str(count), ha="center", va="center", color=text_color,
                        fontsize=16, fontweight="bold")
                pct_label = f"{pct:.0f}%" if pct in (0, 100) else f"{pct:.1f}%"
                ax.text(j, i + .20, pct_label, ha="center", va="center", color=text_color, fontsize=8.5)
        ax.set_yticks(range(3), DISPLAY_LABELS)
        ax.set_xticks(range(3), DISPLAY_LABELS if row == 1 else ["", "", ""])
        ax.tick_params(axis="both", which="both", length=0, pad=6, labelsize=8.5)
        ax.spines[:].set_visible(False)
        ax.set_ylabel("Reference label", fontsize=9, labelpad=8)
        if row == 1:
            ax.set_xlabel("Returned label", fontsize=9, labelpad=8)
        fig.text(x / width, (y + side + .30) / height,
                 f"{chr(97 + index)}  {NAMES[panel['model']]}", fontsize=10.5, weight="bold", color=INK)
        fig.text(x / width, (y + side + .10) / height,
                 "Direct" if panel["method"] == "direct" else "Checklist", fontsize=9, color=INK)
        fig.text((x + side) / width, (y + side + .10) / height,
                 f"Accuracy {panel['accuracy']:.1f}%", ha="right", fontsize=8.5, color=MUTED)
    color_ax = fig.add_axes([6.72 / width, .72 / height, .085 / width, 4.40 / height])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(0, 100), cmap=CMAP), cax=color_ax)
    bar.set_ticks([0, 25, 50, 75, 100])
    bar.ax.tick_params(labelsize=8, length=2, width=.5, pad=2, colors=MUTED)
    bar.set_label("Within-class share (%)", fontsize=8.5, color=MUTED, labelpad=4)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
    assert min(contrasts) >= 4.5
    check = save_figure(fig, "heldout_confusion", output)
    check.update(minimum_cell_text_contrast=min(contrasts), cells=36,
                 common_color_scale=[0, 100], vector_cells=True)
    return check


def signed(value):
    return "0.0" if abs(value) < 1e-10 else f"{value:+.1f}"


def sensitivity_figure(data, output):
    width, height = 7.2, 3.60
    fig = plt.figure(figsize=(width, height))
    bottom, plot_height = .57, 2.54
    labels = [(a[1], a[2]) for a in ANALYSES] + [("Leave one MPA out", "9 subsets, 32 claims each")]
    interval_summary = {}
    for index, (model, left) in enumerate(zip(MODELS, [.20, 3.78])):
        color = COLORS[model]
        ax = fig.add_axes([left / width, bottom / height, 3.22 / width, plot_height / height])
        ax.set(xlim=(-20, 15), ylim=(4.70, -.50))
        ax.set_xticks([-20, -10, 0, 10])
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8.5, length=3, width=.6, colors=MUTED, pad=4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set(color=GRID, linewidth=.7)
        ax.grid(axis="x", color=GRID, linewidth=.6, zorder=0)
        ax.axvline(0, color=MUTED, linestyle=(0, (3, 3)), linewidth=.9, zorder=1)
        for i, (analysis, _, _) in enumerate(ANALYSES):
            value = next(v["percentage_points"] for v in data["sensitivity"]
                         if v["model"] == model and v["analysis"] == analysis)
            point_y = i + .20
            ax.plot([0, value], [point_y, point_y], color=color, linewidth=1.6, alpha=.35, zorder=2,
                    solid_capstyle="round")
            ax.scatter(value, point_y, s=36, color=color, edgecolors="white", linewidth=.7,
                       marker="o" if index == 0 else "s", zorder=3)
            ax.annotate(signed(value), (value, point_y),
                        xytext=(7 if value >= 0 else -7, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center", fontsize=9,
                        color=color, weight="bold", zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .8})
        values = [v["percentage_points"] for v in data["leave_one_mpa_out"] if v["model"] == model]
        assert len(values) == 9
        low, high = min(values), max(values)
        range_y = 4.26
        ax.plot([low, high], [range_y, range_y], color=color, linewidth=1.4, zorder=2)
        same_value = defaultdict(list)
        for value in values:
            same_value[round(value, 8)].append(value)
        for group in same_value.values():
            for j, value in enumerate(group):
                y = range_y + .12 * (j - (len(group) - 1) / 2)
                ax.scatter(value, y, s=10, facecolor="white", edgecolor=color, linewidth=.75, zorder=3)
        place_right = high <= 0
        ax.annotate(f"{signed(low)} to {signed(high)}", (high if place_right else low, range_y),
                    xytext=(8 if place_right else -8, 0), textcoords="offset points",
                    ha="left" if place_right else "right", va="center", fontsize=8.5,
                    color=color, weight="bold", zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .8})
        fig.text(left / width, 3.36 / height, f"{chr(97 + index)}  {NAMES[model]}",
                 fontsize=10.5, weight="bold", color=color)
        interval_summary[model] = [low, high]
    for i, (title, detail) in enumerate(labels):
        y = (bottom + plot_height * (4.70 - (i - .20)) / 5.20) / height
        fig.add_artist(Line2D([.20 / width, 7.00 / width], [y, y],
                              transform=fig.transFigure, color=GRID, linewidth=.65, zorder=4))
        fig.text(.5, y, f"{title}  ({detail})", ha="center", va="center", fontsize=8.8,
                 color=INK, zorder=6,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    fig.text(.5, .08 / height,
             "Checklist - direct accuracy (percentage points)", ha="center", fontsize=9, color=INK)
    check = save_figure(fig, "sensitivity", output)
    check.update(common_x_scale=[-20, 15], ordinary_points=8, exclusion_points=18,
                 exclusion_ranges=interval_summary, intervals_are_confidence_intervals=False,
                 shared_analysis_labels=True, analysis_label_count=5,
                 independent_label_column=False, repeated_analysis_labels=False)
    return check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=MANUSCRIPT / "figures")
    parser.add_argument("--record-dir", type=Path, default=BASE / "04_分析结果/论文图重绘_20260915")
    parser.add_argument("--figure", choices=["all", "confusion", "sensitivity"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.record_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "svg.hashsalt": "ocm-figures-20260915", "axes.unicode_minus": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    data = prepare_data()
    record_path = args.record_dir / "figure_checks.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    checks = {key: previous[key] for key in ["heldout_confusion", "sensitivity"]
              if key in previous and previous.get("source_sha256") == data["source_sha256"]}
    if args.figure in {"all", "confusion"}:
        checks["heldout_confusion"] = confusion_figure(data, args.output_dir)
    if args.figure in {"all", "sensitivity"}:
        checks["sensitivity"] = sensitivity_figure(data, args.output_dir)
    checks.update(source_sha256=data["source_sha256"], source_values_recomputed_and_verified=True,
                  exports=["vector PDF", "editable SVG", "600 dpi PNG"], manuscript_pdf_viewed=False)
    (args.record_dir / "plotted_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    (args.record_dir / "figure_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
