"""Create three complementary manuscript figures from the frozen study records.

Exports editable SVG, vector PDF and direct-rendered 600-dpi PNG previews.
The manuscript PDF is neither opened nor rendered by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Rectangle

from redraw_manuscript_figures import (
    BASE, MANUSCRIPT, MODELS, METHODS, NAMES, INK, MUTED, GRID, TEAL,
    canvas_check, contrast_text, digest,
)

PALE = "#F1F6F7"
PALE_BLUE = "#EEF3F7"
BLUE = "#8BB7C8"
OCHRE = "#DCA871"
GREY = "#E2E8EB"
COPPER = "#A35F2F"
DRAWING_DATA = BASE / "04_分析结果/新增论文图_20260915"
OUTPUT = MANUSCRIPT / "figures"


def prepare_data():
    primary_path = BASE / "04_分析结果/正式研究/正式逐条分析.json"
    summary_path = BASE / "04_分析结果/正式研究/正式研究结果.json"
    additional_path = BASE / "04_分析结果/新增模型对照_20260915/新增解释联合指标.json"
    case_path = BASE / "04_分析结果/法规命题基准/cases_v1.json"
    primary = json.loads(primary_path.read_text())
    summary = json.loads(summary_path.read_text())
    additional = json.loads(additional_path.read_text())
    cases = json.loads(case_path.read_text())
    assert len(primary) == 288 and len(additional["records"]) == 72
    assert additional["human_confirmed"] is True
    assert len(cases) == 72 and len({c["mpa_id"] for c in cases}) == 9
    assert set(Counter(c["mpa_id"] for c in cases).values()) == {8}
    assert Counter(c["split"] for c in cases) == {"pilot": 36, "heldout": 36}
    pages = {p for c in cases for p in c["evidence_ids"]}
    assert len(pages) == 41 and len({p.split(":")[0] for p in pages}) == 11
    case_by_id = {c["case_id"]: c for c in cases}
    groups = []
    for model in MODELS + ["Qwen3-14B"]:
        for method in METHODS:
            is_additional = model == "Qwen3-14B"
            rows = ([r for r in additional["records"] if r["method"] == method]
                    if is_additional else
                    [r for r in primary if r["split"] == "heldout"
                     and r["model"] == model and r["method"] == method])
            assert len(rows) == 36 and len({r["case_id"] for r in rows}) == 36
            categories = [0, 0, 0, 0]
            membership = [[], [], [], []]
            for r in rows:
                label = bool(r["correct"])
                compatible = r["rationale_status"] == "consistent"
                aligned = r["alignment"] == "aligned"
                joint = label and compatible
                aligned_joint = joint and aligned
                assert label == (r["prediction"] == case_by_id[r["case_id"]]["gold_label"])
                joint_key = "joint_compatible" if is_additional else "joint_consistent"
                aligned_key = joint_key + "_aligned"
                assert joint == r[joint_key] and aligned_joint == r[aligned_key]
                index = 0 if aligned_joint else 1 if joint else 2 if label else 3
                categories[index] += 1
                membership[index].append(r["case_id"])
            if is_additional:
                archived = next(g for g in additional["groups"] if g["method"] == method)
                expected = [archived["joint_compatible_aligned_count"],
                            archived["joint_compatible_count"], archived["label_correct"]]
            else:
                archived = next(g for g in summary["splits"]["heldout"]["groups"]
                                if g["model"] == model and g["method"] == method)
                expected = [archived["joint_consistent_aligned"],
                            archived["joint_consistent"], archived["correct"]]
            observed = [categories[0], sum(categories[:2]), sum(categories[:3])]
            assert observed == expected, (model, method, observed, expected)
            groups.append(dict(model=NAMES.get(model, model), method=method, n=len(rows),
                               counts=categories, case_ids=membership,
                               aligned_joint=observed[0], joint=observed[1], label_correct=observed[2]))
    case = case_by_id["STI08"]
    assert case["gold_label"] == "INSUFFICIENT" and case["assessment_date"] == "2024-05-04"
    return dict(
        groups=groups,
        partition=["Aligned joint", "Joint without confirmed alignment",
                   "Correct label outside joint", "Incorrect label"],
        category_definition=["L and E and A", "L and E and not A", "L and not E", "not L"],
        criteria=dict(L="Label matches reference", E="Explanation coded consistent",
                      A="Explanation coded aligned"),
        framework=dict(contextual_documents=12, contextual_pages=139,
                       input_documents=11, distinct_input_pages=41, mpas=9,
                       claims=72, pilot=36, heldout=36, primary_responses=288,
                       additional_responses=72, assessed_explanations=360),
        worked_case={k: case[k] for k in ["case_id", "mpa_id", "assessment_date", "facts",
                                         "claim", "gold_label", "gold_required_evidence"]},
        conceptual_scope="Proposed review structure illustrated by a constructed case; not a field evaluation",
        source_sha256={str(p.relative_to(BASE)): digest(p)
                       for p in [primary_path, summary_path, additional_path, case_path]},
    )


def sheet(width, height):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def text(ax, x, y, label, size=9, color=INK, weight="normal", ha="left", va="top", **kwargs):
    return ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, linespacing=1.32, **kwargs)


def rect(ax, x, y, w, h, face=PALE, edge=GRID, lw=.65, zorder=1, linestyle="solid"):
    patch = Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                      linewidth=lw, zorder=zorder, linestyle=linestyle)
    ax.add_patch(patch)
    return patch


def line(ax, points, color=GRID, lw=.8, **kwargs):
    return ax.plot([p[0] for p in points], [p[1] for p in points],
                   color=color, linewidth=lw, solid_capstyle="butt", **kwargs)


def arrow(ax, start, end, color=MUTED, lw=.8, scale=7):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale,
                            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                            zorder=2)
    ax.add_patch(patch)


def group_box(ax, x, y, width, height, title):
    rect(ax, x, y, width, height, "white", "#B8CDD3", .8)
    rect(ax, x, y + height - .34, width, .34, PALE, "none", 0)
    text(ax, x + width / 2, y + height - .17, title, 10,
         TEAL, "bold", ha="center", va="center")


def save_figure(fig, stem, title):
    check = canvas_check(fig)
    metadata = dict(Title=title, Author="", Creator="Matplotlib",
                    CreationDate=None, ModDate=None)
    fig.savefig(OUTPUT / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(OUTPUT / f"{stem}.pdf", metadata=metadata)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=600)
    check["output_sha256"] = {suffix: digest(OUTPUT / f"{stem}.{suffix}")
                              for suffix in ["svg", "pdf", "png"]}
    check["png_dpi"] = 600
    check["svg_text_editable"] = True
    plt.close(fig)
    return check


def framework_figure(data):
    fig, ax = sheet(7.2, 3.42)
    for x, title in [(.06, "Sources and cases"), (2.51, "Paired model tests"),
                     (4.96, "Assessment criteria")]:
        group_box(ax, x, .07, 2.21, 3.25, title)

    # A single input-to-assessment flow, with two arrows between whole stages.
    arrow(ax, (2.29, 1.73), (2.49, 1.73), TEAL, scale=6.5)
    arrow(ax, (4.74, 1.73), (4.94, 1.73), TEAL, scale=6.5)
    text(ax, 1.165, 2.78, "Contextual corpus", 9.3, weight="bold", ha="center")
    text(ax, 1.165, 2.55, "12 documents · 139 pages", 9, ha="center")
    text(ax, 1.165, 2.24, "Model inputs", 9.3, weight="bold", ha="center")
    text(ax, 1.165, 2.01, "11 documents · 41 selected pages", 8.5, MUTED, ha="center")
    line(ax, [(.20, 1.78), (2.13, 1.78)], "#B8CDD3", .7, linestyle=(0, (3, 2)))
    text(ax, 1.165, 1.61, "72 constructed claims", 10, TEAL, "bold", ha="center")
    text(ax, 1.165, 1.35, "9 MPAs × 8 claims", 9, ha="center")
    for x, label in [(.20, "Pilot  36"), (1.22, "Held-out  36")]:
        rect(ax, x, .84, .91, .32, PALE, "none", 0)
        text(ax, x + .455, 1.00, label, 8.6, TEAL, "bold", ha="center", va="center")
    text(ax, 1.165, .63, "Date · scope · facts · claim", 8.5, MUTED, ha="center")
    text(ax, 1.165, .39, "Selected source pages", 8.5, MUTED, ha="center")

    rect(ax, 2.65, 2.38, 1.93, .49, PALE_BLUE, "none", 0)
    text(ax, 3.615, 2.76, "Direct  |  Checklist", 10, weight="bold", ha="center")
    text(ax, 3.615, 2.52, "Same facts and source pages", 8.3, MUTED, ha="center")
    rect(ax, 2.65, 1.45, 1.93, .79, "white")
    text(ax, 3.615, 2.10, "Primary comparison", 9.3, TEAL, "bold", ha="center")
    text(ax, 3.615, 1.85, "Qwen3-4B · Phi-4-mini", 9.4, weight="bold", ha="center")
    text(ax, 3.615, 1.61, "72 × 2 models × 2 prompts = 288", 8.15, ha="center")
    rect(ax, 2.65, .70, 1.93, .61, "white", "#B8CDD3", .7, linestyle=(0, (3, 2)))
    text(ax, 3.615, 1.17, "Additional · Qwen3-14B", 9.3, TEAL, "bold", ha="center")
    text(ax, 3.615, .91, "36 held-out × 2 prompts = 72", 8.5, ha="center")
    text(ax, 3.615, .53, "360 responses", 10.2, TEAL, "bold", ha="center")
    text(ax, 3.615, .28, "Label · source IDs · explanation", 8.2, MUTED, ha="center")

    for y, title, body in [
        (2.25, "L  Label agreement", "Matches the reference label"),
        (1.52, "E  Explanation content", "No definite substantive error"),
        (.79, "A  Label alignment", "Explanation supports its label"),
    ]:
        rect(ax, 5.10, y, 1.93, .62, PALE, "none", 0)
        text(ax, 6.065, y + .47, title, 9.5, weight="bold", ha="center")
        text(ax, 6.065, y + .22, body, 8.35, MUTED, ha="center")
    rect(ax, 5.10, .18, 1.93, .47, TEAL, TEAL, 0)
    text(ax, 6.065, .515, "Joint: L and E", 9.3, "white", "bold", ha="center", va="center")
    text(ax, 6.065, .315, "Aligned joint: L, E and A", 8.5, "white", ha="center", va="center")
    check = save_figure(fig, "study_framework", "Study design and evaluation framework")
    check.update(subpanel_labels=[], arrow_count=2,
                 arrow_meaning="Case inputs to paired model tests; responses to assessment",
                 source_reference_answers_excluded_from_model_inputs=True)
    return check


def joint_figure(data):
    fig, ax = sheet(3.5, 4.43)
    x0, x1 = .17, 3.33
    scale = (x1 - x0) / 36
    colors = [TEAL, BLUE, OCHRE, GREY]
    centers = [3.40, 3.00, 2.36, 1.96, 1.32, .92]
    text(ax, 1.75, .20, "Responses (n = 36 per condition)", 9.3,
         weight="bold", ha="center", va="center")
    line(ax, [(x0, .68), (x1, .68)], GRID, .7)
    for value in range(0, 37, 12):
        x = x0 + value * scale
        line(ax, [(x, .64), (x, .72)], GRID, .7)
        text(ax, x, .48, str(value), 8.2, MUTED, ha="center", va="center")
    contrasts = []
    for i, (group, y) in enumerate(zip(data["groups"], centers)):
        if i % 2 == 0:
            text(ax, x0, y + .41, group["model"], 9.5, TEAL,
                 "bold", ha="left", va="center")
        text(ax, x0, y + .20, "Direct" if group["method"] == "direct" else "Checklist",
             8.3, MUTED, ha="left", va="center")
        left = x0
        for value, color in zip(group["counts"], colors):
            if not value:
                continue
            width = value * scale
            rect(ax, left, y - .103, width, .206, color, "white", .6)
            ink, contrast = contrast_text(to_rgb(color))
            contrasts.append(contrast)
            text(ax, left + width / 2, y, str(value), 8.5, ink, "bold", ha="center", va="center")
            left += width
        assert abs(left - x1) < 1e-9
    for y in [2.87, 1.83]:
        line(ax, [(x0, y), (x1, y)], GRID, .6, linestyle=(0, (3, 2)))
    handles = [Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none") for color in colors]
    # Centre each row independently while preserving the left-to-right segment order.
    legends = []
    for indices, y in [((0, 1), 4.28), ((2, 3), 4.065)]:
        legend = ax.legend([handles[i] for i in indices],
                           [data["partition"][i] for i in indices],
                           loc="center", bbox_to_anchor=(1.75, y), bbox_transform=ax.transData,
                           ncol=2, frameon=False, fontsize=8.1, labelcolor=INK,
                           handlelength=.9, handleheight=.8, handletextpad=.5,
                           columnspacing=.9, borderpad=0, borderaxespad=0)
        if not legends:
            ax.add_artist(legend)
        legends.append(legend)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_centre_offsets = [abs(legend.get_window_extent(renderer).x0
                                 + legend.get_window_extent(renderer).width / 2
                                 - fig.bbox.width / 2) for legend in legends]
    assert max(legend_centre_offsets) < .1
    check = save_figure(fig, "joint_assessment", "Held-out responses under nested reliability criteria")
    check["minimum_segment_text_contrast"] = min(contrasts)
    check["bar_counts"] = [g["counts"] for g in data["groups"]]
    check.update(subpanel_labels=[], arrow_count=0, target_width="single column",
                 legend_position="top centre, two rows in segment order",
                 legend_row_centre_offsets_px=legend_centre_offsets,
                 axis_position="bottom, ticks above the centred axis title",
                 category_labels_position="left aligned with the bar origin")
    return check


def evidence_figure(data):
    fig, ax = sheet(7.2, 3.44)
    group_box(ax, .07, .09, 2.40, 3.25, "Evidence and scope")
    group_box(ax, 2.82, .09, 4.31, 3.25, "Proposed permit review")

    # Evidence types connect to their corresponding case entries, not to one another.
    line(ax, [(2.33, 2.54), (2.60, 2.54), (2.60, 2.02), (3.01, 2.02)], TEAL, .8)
    line(ax, [(2.33, 1.61), (2.70, 1.61), (2.70, 1.10), (3.01, 1.10)],
         "#447F98", .8, linestyle=(0, (3, 2)))
    for y, color, title, body in [
        (2.19, TEAL, "Regulations", "Obligations and permissions\nActor · activity · zone · date"),
        (1.26, "#447F98", "Individual records", "Individual authorisation\nHolder · activity · period"),
        (.33, COPPER, "Observations", "Conduct and ecological change\nActivity and outcome measurements"),
    ]:
        rect(ax, .21, y, 2.12, .70, "white", color, .7)
        rect(ax, .21, y, .035, .70, color, color, 0)
        text(ax, .36, y + .56, title, 9.7, color, "bold")
        text(ax, .36, y + .31, body, 8.4, MUTED)

    text(ax, 4.975, 2.83, "STI08 · Stilbaai · 4 May 2024", 9, weight="bold", ha="center")
    text(ax, 4.975, 2.63, "Claim: permit issued to the SCUBA-diving business", 8.6, MUTED, ha="center")
    rect(ax, 3.01, 1.66, 2.02, .73, PALE, TEAL, .7)
    text(ax, 4.02, 2.23, "Rule established", 9.5, TEAL, "bold", ha="center")
    text(ax, 4.02, 1.97, "Business operation\nrequires a permit", 8.9, ha="center")
    rect(ax, 3.01, .74, 2.02, .73, PALE_BLUE, "#447F98", .8, linestyle=(0, (3, 2)))
    text(ax, 4.02, 1.31, "Record not supplied", 9.5, "#447F98", "bold", ha="center")
    text(ax, 4.02, 1.05, "No individual permit or\nregistry entry supplied", 8.9, ha="center")

    # The supplied rule and identified evidence gap jointly inform the determination.
    line(ax, [(5.03, 2.02), (5.19, 2.02), (5.19, 1.10)], TEAL, .8)
    line(ax, [(5.03, 1.10), (5.19, 1.10)], "#447F98", .8, linestyle=(0, (3, 2)))
    arrow(ax, (5.19, 1.65), (5.42, 1.65), TEAL, scale=6.5)
    rect(ax, 5.42, 1.12, 1.56, 1.06, TEAL, TEAL, 0)
    text(ax, 6.20, 1.94, "INSUFFICIENT", 10.7, "white", "bold", ha="center")
    text(ax, 6.20, 1.58, "Issuance remains\nunresolved", 9, "white", ha="center")
    arrow(ax, (6.20, 1.12), (6.20, .64), TEAL, scale=6.5)
    rect(ax, 3.01, .23, 3.97, .37, PALE, "none", 0)
    text(ax, 4.995, .415, "Next step: request the individual permit or registry entry", 8.5,
         TEAL, ha="center", va="center")
    check = save_figure(fig, "evidence_review", "Evidence boundaries and a proposed management review")
    check.update(subpanel_labels=[], arrow_count=2,
                 arrow_meaning="Rule and evidence gap to reviewed determination; determination to follow-up",
                 dashed_lines="Individual authorisation record not supplied",
                 observations_not_used_to_determine_permit_issuance=True)
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "framework", "joint", "evidence"], default="all")
    args = parser.parse_args()
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.unicode_minus": True,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DRAWING_DATA.mkdir(parents=True, exist_ok=True)
    data = prepare_data()
    (DRAWING_DATA / "plotted_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    check_path = DRAWING_DATA / "figure_checks.json"
    checks = json.loads(check_path.read_text()) if check_path.exists() else {}
    for name, function in [("framework", framework_figure), ("joint", joint_figure), ("evidence", evidence_figure)]:
        if args.figure in {"all", name}:
            checks[name] = function(data)
    check_path.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"figures": list(checks), "bar_counts": [g["counts"] for g in data["groups"]],
                      "checks": str(check_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
