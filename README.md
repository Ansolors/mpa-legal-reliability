# MPA legal interpretation reliability

Data and code accompanying **Reliability of large language models in interpreting South African marine protected area regulations**.

Authors: Yifan Qi, Rence Shuai, Yingze Zhao and Yulin Miao.

## Study materials

The benchmark contains 72 constructed claims across nine South African marine protected areas (MPAs). Cases 01–04 in each MPA form the 36-claim pilot; cases 05–08 form the 36-claim held-out evaluation. The primary experiment comprises 288 responses from Qwen3-4B and Phi-4-mini under direct and checklist prompting. The additional Qwen3-14B comparison contributes 72 responses on the held-out cases. In machine-readable records, `structured` denotes the checklist condition.

The repository includes all 12 legal documents, extracted source pages, the case inventory and reference labels, exact prompts and formatted model inputs, raw outputs, explanation codes, reported results, model configurations and file fingerprints, and inference and analysis scripts. Model weights are obtained separately from the public repositories below.

## Directory guide

| Location | Contents |
|---|---|
| `03_法规与政策/` | Original government legal documents |
| `04_分析结果/法规命题基准/` | Case export, source pages, publication checksums and original freeze manifest |
| `04_分析结果/模型先导实验/` | Primary prompts, scoring rules, 288 raw responses and run metadata |
| `04_分析结果/正式研究/` | Primary explanation codes, joined records, results, sensitivity analyses, legal-source manifest and software environment |
| `04_分析结果/新增模型对照_20260915/` | Additional protocol, 72 raw responses, final explanation codes and results |
| `02_原始数据/` | Model configurations, repository listings and revision metadata |
| `05_分析代码/` | Inference, scoring, analysis and figure-generation scripts |
| `06_验证日志/新增模型对照_20260915/下载校验.json` | Archived model download locations and file fingerprints |

The case inventory is a publication export with unchanged claims, facts, labels and splits. `publication_manifest.json` records the checksums of the distributed case and evidence files; the original freeze manifest is retained as an archival record and identifies the pre-experiment files. Raw response files and experimental protocols retain their original bytes. The `explanation_codes.json` files contain the final coding values used in the manuscript. Source-document names and page IDs are retained to connect the cases, prompts and legal evidence. Chinese directory names preserve the paths used by the original scripts; text files use UTF-8.

## Analyse the archived responses

Use Python 3.10 or later. The scoring and numerical-analysis scripts use the standard library; no model download is needed. Run from the repository root, in the following order:

```sh
python3 05_分析代码/analyse_formal.py
python3 05_分析代码/analyse_joint_sensitivity.py
python3 05_分析代码/analyse_comparator.py
python3 05_分析代码/analyse_comparator_explanations.py
python3 05_分析代码/make_confirmed_comparator_tables.py --output-dir generated/tables
```

Analysis commands write derived results to the corresponding analysis directories. The final command exports the combined explanation and sensitivity tables. Primary records and additional-model records remain separate.

`consistent` denotes an explanation with no identified definite substantive error; `substantive_error` denotes an erroneous explanation; `indeterminate` denotes an ambiguous explanation. Alignment is assessed separately as `aligned`, `misaligned` or `indeterminate`. The joint criterion requires a correct label and a `consistent` explanation; the aligned joint criterion additionally requires `aligned`. Ambiguous explanations remain in the denominator.

## Figures

```sh
python3 -m pip install -r requirements-analysis.txt
python3 05_分析代码/redraw_manuscript_figures.py
python3 05_分析代码/make_additional_manuscript_figures.py --figure all
```

Figures are exported to `generated/figures/`. Rendering uses Arial when available; another installed sans-serif font may be substituted by Matplotlib.

## Inference and model sources

The archived environment is recorded in `04_分析结果/正式研究/正式研究环境.json`. Inference used MLX on Apple silicon, 4-bit models, greedy decoding, seed 20260910 and a 256-token generation budget. Qwen3-14B used its non-thinking chat-template mode. Exact instructions and complete inputs are included in the protocol files and raw response records.

- [Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)
- [Phi-4-mini-instruct-4bit](https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit)
- [Qwen3-14B-4bit](https://huggingface.co/mlx-community/Qwen3-14B-4bit)

`download_models.py` and `download_comparator.py` retrieve model files into `02_原始数据/模型权重/`; revision metadata and fingerprints identify the archived implementations. `run_pilot.py` contains the common prompt and both primary prompting conditions, and `run_comparator.py` contains the additional-model configuration. Run inference only in a separate working copy if producing new observations; the archived response files should be retained unchanged. Inference dependencies are listed in `requirements-inference.txt`.

## Scope and reuse

These are constructed document-grounded cases, not observations of individual operators or a determination of compliance with all South African law. Legal-source URLs and file fingerprints are listed in `04_分析结果/正式研究/法源清单.json`. Government source documents and third-party model configurations retain their respective original terms; model weights are not redistributed.

Use the current `main` branch for the publication materials and record the commit identifier when citing a specific snapshot. Citation metadata are provided in `CITATION.cff`.
