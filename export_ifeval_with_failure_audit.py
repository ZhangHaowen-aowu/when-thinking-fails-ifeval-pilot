#!/usr/bin/env python3
"""Audited IFEval export with a pre-declared policy for incomplete generations.

Retains all pre-selected prompts. For a prompt-mode without ANY complete generation,
counts the generation as an end-to-end failure by exporting an empty scored answer.
Additionally exports the earliest actual partial response for a sensitivity check.
Does not call models, alter prompts, or overwrite raw generations.
"""
import argparse, json
from collections import Counter
from pathlib import Path

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def dump(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def parse_cot(raw):
    a = raw.find("THINK:")
    b = raw.find("ANSWER:", a + 6) if a >= 0 else -1
    if a < 0 or b < 0:
        return raw.strip(), False
    return raw[b + len("ANSWER:"):].strip(), True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples_60.jsonl")
    p.add_argument("--raw", default="data/generations.jsonl")
    p.add_argument("--out-dir", default="data/final_eval_inputs")
    args = p.parse_args()

    samples = load(args.samples)
    records = load(args.raw)
    if len(set(str(x["key"]) for x in samples)) != len(samples):
        raise ValueError("Duplicate keys in selected sample file")
    source_ids = {str(x["key"]) for x in samples}
    by = {}
    for r in records:
        key = (str(r.get("key")), r.get("mode"))
        if key[0] in source_ids and key[1] in {"direct", "cot"}:
            by.setdefault(key, []).append(r)

    root = Path(args.out_dir)
    outputs = {k: [] for k in ["direct", "cot_final", "cot_full", "direct_partial_raw"]}
    audit = []
    for sample in samples:
        k = str(sample["key"])
        selected = {}
        for mode in ("direct", "cot"):
            candidates = by.get((k, mode), [])
            if not candidates:
                raise ValueError(f"Missing all runs: {k}/{mode}")
            models = {r.get("model") for r in candidates if r.get("ok")}
            if len(models) > 1:
                raise ValueError(f"Mixed model IDs for {k}/{mode}: {models}")
            complete = []
            for i, r in enumerate(candidates, 1):
                raw = r.get("raw") or ""
                answer, fmt = parse_cot(raw) if mode == "cot" else (raw.strip(), True)
                if (r.get("ok") is True and r.get("finish_reason") != "length"
                        and bool(answer.strip())):
                    complete.append((i, r, answer, fmt))
            if complete:
                index, r, answer, fmt = complete[0]
                is_complete = True
            else:
                # Deterministic, reproducible fallback; we do NOT disguise a
                # truncated completion as a complete answer.
                index, r = next(
                    ((i, x) for i, x in enumerate(candidates, 1) if x.get("ok") and (x.get("raw") or "").strip()),
                    (None, None))
                if r is None:
                    raise ValueError(f"No usable raw text at all: {k}/{mode}")
                raw = r.get("raw") or ""
                answer, fmt = parse_cot(raw) if mode == "cot" else (raw.strip(), True)
                is_complete = False
            selected[mode] = {"r": r, "answer": answer, "complete": is_complete}
            audit.append({
                "key": k, "mode": mode, "selected_attempt": index,
                "num_attempts": len(candidates),
                "complete": is_complete,
                "finish_reason": r.get("finish_reason"),
                "model": r.get("model"), "cot_format_ok": fmt if mode == "cot" else None,
                "fallback_used": mode == "cot" and not fmt,
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
                "scoring_policy": ("actual_complete_generation" if is_complete
                                   else "empty_answer_counted_as_failure"),
                "raw_partial_retained_for_audit": not is_complete,
            })

        direct, cot = selected["direct"], selected["cot"]
        # Primary analysis: a generation that never completed is not a valid
        # final response. Count it as failure for ALL instruction constraints.
        d_scored = direct["answer"] if direct["complete"] else ""
        c_final = cot["answer"] if cot["complete"] else ""
        c_full = (cot["r"].get("raw") or "") if cot["complete"] else ""
        d_partial = direct["answer"]

        for name, val in [
            ("direct", d_scored),
            ("cot_final", c_final),
            ("cot_full", c_full),
            ("direct_partial_raw", d_partial),
        ]:
            outputs[name].append({"prompt": sample["prompt"], "response": val})

    dump(root / "input_data.jsonl", samples)
    for mode, data in outputs.items():
        dump(root / mode / "responses.jsonl", data)
    dump(root / "selected_runs_audit.jsonl", audit)

    c = Counter((a["mode"], a["finish_reason"], a["complete"]) for a in audit)
    incomplete = [a for a in audit if not a["complete"]]
    noformat = [a for a in audit if a["fallback_used"]]
    print(f"Exported {len(samples)} prompts to {root}")
    print("Incomplete generation(s) counted as failure:", [(a["key"], a["mode"], a["finish_reason"]) for a in incomplete])
    print("CoT missing-marker fallback:", [a["key"] for a in noformat])
    print("Primary: direct / cot_final / cot_full.")
    print("Sensitivity: direct_partial_raw, preserving the earliest unfinished text.")
    print("Raw generations unchanged. Audit: selected_runs_audit.jsonl")

if __name__ == "__main__":
    main()
