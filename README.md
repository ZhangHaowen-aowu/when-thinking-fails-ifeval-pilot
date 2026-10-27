# When Thinking Fails — 60-Prompt IFEval Pilot

Small-scale exploratory reproduction and evaluation-boundary sensitivity analysis of [Li et al., *When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs* (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/706338a08f9378b708f21cbf5686e617-Abstract-Conference.html).

This repository contains **actual** frozen inputs, generation logs, official IFEval scores, exception audit, and analysis code from a 60-prompt pilot prepared for HKU COMP7404 (Group 44). It is **not** a complete reproduction of the original paper's full model or benchmark suite. The repo maintainer should verify the academic attribution, authorship and AI-use disclosure before publishing.

## Results (verified against included scoring JSONL)

Instruction-level accuracy, 87 evaluated constraints across 60 preselected prompts:

| Evaluation condition | Strict | Loose |
| --- | ---: | ---: |
| Direct, final answer | 69/87 = **79.31%** | 69/87 = **79.31%** |
| CoT, parsed final answer | 61/87 = **70.11%** | 63/87 = **72.41%** |
| CoT, full visible text | 54/87 = **62.07%** | 65/87 = **74.71%** |

**Main comparison:** CoT Final − Direct = **−6.90 percentage points** (loose instruction-level; paired prompt-bootstrap 95% interval **[−16.05, +2.22] pp**). The interval includes zero; this pilot is **not evidence of a statistically significant decrease**. Direction matches the paper's reported Qwen2.5-7B-Instruct result (63.6% Direct vs 57.7% CoT), but absolute levels differ due to dataset size/selection, inference environment and handling rules.

**Paired prompt transitions, loose:** both pass 35; both fail 12; Direct only 9; CoT only 4.

**Sensitivity to evaluation boundary:** scoring the entire visible CoT text rather than only `ANSWER:` changes strict accuracy from 70.11% to 62.07%, but loose accuracy from 72.41% to 74.71%. Full-trace scoring is an **additional method-sensitivity test, not the source paper's main evaluation protocol**. Loose IFEval tests multiple adjusted text variants; these results do not prove an error in the original paper.

## Experimental setup

- **Data:** IFEval `input_data.jsonl` (541 prompts), included for reproducibility; `data/samples_60.jsonl` is the frozen 60-prompt sample drawn by Python `random.Random(42).shuffle()` (not stratified). **87 constraints** in selected prompts.
- **Model:** `Qwen/Qwen2.5-7B-Instruct` served via the SiliconFlow OpenAI-compatible API. Provider-side exact model snapshot/version is not established in the archived records.
- **Inference:** `temperature=0`, one visible `user` message per call. Explicit CoT prompt is the template from the original paper's released `templates.py` (see upstream repository below). Direct uses unmodified instruction. Maximum output budgets used in successive initial/retry passes were 2048, 4096 and 8192 tokens.
- **Generation history:** `data/generations.jsonl` includes **130 saved attempts** (including retries) for 120 prompt-mode combinations. Do **not** interpret retry history as 130 distinct experimental prompts.
- **Response extraction:** CoT responses with `THINK:` / `ANSWER:` are split into visible thinking and final answer. When those markers are absent, the entire text is used as answer (a fallback analogous to the authors' released helper), flagged in the audit.
- **Incomplete output handling:** the Direct output for prompt key `3567` repeated until the token budget and never completed; for the primary analysis its response is scored as empty/failure for all constraints. An alternative (`direct_partial_raw`) scores the earliest actual incomplete text only for sensitivity analysis.
- **Scoring:** copied Google official IFEval implementation, instruction-level strict and loose, using each method's selected response text. `PYTHONUTF8=1` is recommended on Windows for Unicode text.

### Five flagged prompts and sensitivity check

Four CoT runs had missing format markers: `2549`, `2531`, `2622`, `1954`. One Direct run was incomplete: `3567`. The **55-prompt subset excluding all five flags** retains 80 constraints: Direct loose **64/80 = 80.00%**, CoT Final loose **59/80 = 73.75%** (**−6.25 pp**, paired bootstrap 95% interval **[−14.94, +2.44] pp**). This is a robustness check, not the primary result. No prompts were silently dropped from the primary 60-prompt analysis.

## Repository map

```text
run_experiment.py                       Initial generation/prepare implementation (historical)
run_experiment_v2.py                    Updated generation/resume/parse behavior
export_ifeval_with_failure_audit.py     Final audit-preserving exporter (used for published data)
analyze_ifeval_paired.py                Paired transitions + prompt-level bootstrap
data/samples_60.jsonl                   Frozen 60-prompt sample
input_data.jsonl                        IFEval full 541-prompt input file
data/generations.jsonl                  130 raw attempts, including retries
data/final_eval_inputs/input_data.jsonl Frozen prompts for evaluator
data/final_eval_inputs/{mode}/responses.jsonl         Responses for scoring
data/final_eval_inputs/{mode}/eval_results_*.jsonl    Actual official evaluator outputs
data/final_eval_inputs/selected_runs_audit.jsonl     Exact selected run & failure policy
data/final_eval_inputs/paired_analysis_summary.json  Summary and confidence intervals
instruction_following_eval/             Local copy of essential upstream Google evaluator files
THIRD_PARTY_NOTICES.md                  Original-paper, benchmark and evaluator attribution
AI_USE_DISCLOSURE.md                    Disclosure draft for maintainer verification
```

Modes are `direct`, `cot_final`, `cot_full`, and the sensitivity condition `direct_partial_raw`.

## Verify the published statistics (no API calls)

Python 3.10+:

```powershell
python .\analyze_ifeval_paired.py --folder data/final_eval_inputs
```

This reads the already-produced official IFEval score files and prints counts and confidence intervals. The script writes/replaces `paired_analysis_summary.json` with the same statistical definitions.

To **re-score the archived responses** (no API calls):

```powershell
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
$env:PYTHONUTF8 = "1"
foreach ($mode in @("direct", "cot_final", "cot_full", "direct_partial_raw")) {
    python -m instruction_following_eval.evaluation_main `
      --input_data=data/final_eval_inputs/input_data.jsonl `
      --input_response_data="data/final_eval_inputs/$mode/responses.jsonl" `
      --output_dir="data/final_eval_inputs/$mode"
}
```

To **re-export selected responses from raw logs** (no API calls):

```powershell
python .\export_ifeval_with_failure_audit.py --out-dir data/final_eval_inputs
```

This overwrites only derived export files and the audit file, not the raw generation log. The whole-trace sensitivity condition uses the full visible model text including literal prompt-format markers.

To **run new generations** (uses paid model API; use a fresh output path to preserve provided results):

```powershell
$env:OPENAI_API_KEY = Read-Host 'SiliconFlow API key'
$env:OPENAI_BASE_URL = 'https://api.siliconflow.cn/v1'
$env:OPENAI_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
python .\run_experiment_v2.py generate --mode both --out data/new_generations.jsonl
```

**Important:** New calls may differ due to provider-side model revisions or randomness outside the client control. Neither `temperature=0` nor a model ID guarantees bitwise identical results. The archived generation log is the experiment's primary provenance.

## Interpretation and limitations

- 60 random prompts from IFEval, one model, one API provider; smaller and not matched to the original full benchmark.
- Loose metric variants can admit some constraints even when whole raw output seems noncompliant; don't interpret full-trace scores as actual compliance or as evidence the paper's main evaluator was wrong.
- Output formatting can fail; four CoT responses were scored using the missing-marker fallback rather than demonstrably separated reasoning and final answer.
- One Direct generation was truncated due to repeated output; primary policy counts incomplete generation as failure. Treat as a methodological choice, not an upstream paper replication detail.
- Paired prompt bootstrap resamples prompts (not individual constraints), 10,000 draws, seed 42; CI crosses zero. Per-category subgroup rates are exploratory and generally small.
- Backend implementation, exact model snapshot, and any changes between service revisions are not documented in this archive.

## Citations / attribution

- Li et al., *When Thinking Fails* (NeurIPS 2025): https://proceedings.neurips.cc/paper_files/paper/2025/hash/706338a08f9378b708f21cbf5686e617-Abstract-Conference.html
- Authors' CoT template and evaluation helpers: https://github.com/amazon-science/when-thinking-fails-RLLM-if-evaluation
- IFEval benchmark and scoring code, Zhou et al. (2023): https://github.com/google-research/google-research/tree/master/instruction_following_eval

See `THIRD_PARTY_NOTICES.md`. The project maintainer must verify course authorship, group information, AI disclosure and desired licensing before public release.
