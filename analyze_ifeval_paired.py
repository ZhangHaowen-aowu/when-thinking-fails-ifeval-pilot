#!/usr/bin/env python3
"""Paired, audit-aware summary for 60-item IFEval pilot.
Reads existing official evaluator outputs; does NOT call a model or alter them.
Usage: python analyze_ifeval_paired.py --folder data/final_eval_inputs
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding='utf-8') as fh:
        return [json.loads(x) for x in fh if x.strip()]


def pct(a, b):
    return f'{100*a/b:.2f}%' if b else 'N/A'


def report_difference(selected_prompts, direct, cot, title, rng_seed=42):
    keys = [p for p in selected_prompts if p in direct and p in cot]
    pairs = []
    switches = Counter()
    per_id = defaultdict(lambda: [0,0,0,0])
    for prompt in keys:
        d, c = direct[prompt], cot[prompt]
        d_flags, c_flags = d['follow_instruction_list'], c['follow_instruction_list']
        labels = d['instruction_id_list']
        assert labels == c['instruction_id_list'], f'Instruction order mismatch: {prompt[:60]}'
        d_good, c_good = sum(d_flags), sum(c_flags)
        pairs.append((d_good, c_good, len(d_flags)))
        ds = bool(d['follow_all_instructions'])
        cs = bool(c['follow_all_instructions'])
        switches['prompt:both_pass' if ds and cs else
                 'prompt:direct_only' if ds and not cs else
                 'prompt:cot_only' if cs and not ds else
                 'prompt:both_fail'] += 1
        for lab, df, cf in zip(labels, d_flags, c_flags):
            switches['constraint:both_pass' if df and cf else
                     'constraint:direct_only' if df and not cf else
                     'constraint:cot_only' if cf and not df else
                     'constraint:both_fail'] += 1
            grp = lab.split(':')[0]
            row = per_id[grp]
            row[0] += int(df)
            row[1] += int(cf)
            row[2] += 1
            row[3] += int(bool(df) != bool(cf))

    dt = sum(x[0] for x in pairs)
    ct = sum(x[1] for x in pairs)
    total = sum(x[2] for x in pairs)
    delta_pp = (ct - dt) / total * 100
    # Nonparametric paired bootstrap clustered at the prompt, not instruction.
    # Descriptive uncertainty; small samples and heterogeneous constraints.
    rng = random.Random(rng_seed)
    samples = []
    for _ in range(10000):
        draws = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        den = sum(x[2] for x in draws)
        samples.append(100 * sum(x[1] - x[0] for x in draws) / den)
    samples.sort()
    ci = (samples[249], samples[9749])

    print('\n===', title, '===')
    print('Prompts:', len(keys), '| Constraints:', total)
    print('Direct:', f'{dt}/{total}', pct(dt,total), '| CoT final:', f'{ct}/{total}', pct(ct,total))
    print(f'CoT minus Direct: {delta_pp:+.2f} percentage points')
    print(f'Paired bootstrap 95% interval (resample prompts): [{ci[0]:+.2f}, {ci[1]:+.2f}] pp')
    print('Prompt transitions:', {k.split(':')[1]:v for k,v in switches.items() if k.startswith('prompt:')})
    print('Constraint transitions:', {k.split(':')[1]:v for k,v in switches.items() if k.startswith('constraint:')})
    print('Category (direct, CoT, checked):')
    for grp, (dv,cv,den,discord) in sorted(per_id.items()):
        print(f'  {grp:20} {dv:>2}/{den:<2} -> {cv:>2}/{den:<2} ({100*(cv-dv)/den:+6.1f} pp; paired changes={discord})')
    return {'n_prompts':len(keys),'n_constraints':total,'direct_correct':dt,'cot_correct':ct,
            'delta_pp':round(delta_pp,3),'bootstrap_95_ci_pp':[round(c,3) for c in ci],
            'prompt_transitions':{k.split(':')[1]:v for k,v in switches.items() if k.startswith('prompt:')},
            'constraint_transitions':{k.split(':')[1]:v for k,v in switches.items() if k.startswith('constraint:')}}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--folder', default='data/final_eval_inputs')
    args = pa.parse_args()
    root = Path(args.folder)
    inp = read_jsonl(root/'input_data.jsonl')
    audit = read_jsonl(root/'selected_runs_audit.jsonl')
    prompts = [i['prompt'] for i in inp]
    ids = {str(i['key']):i['prompt'] for i in inp}
    excluded_ids = set()
    print('=== Generation audit ===')
    for a in audit:
        if (not a.get('complete', True)) or a.get('fallback_used', False):
            print('Check:', a['key'],a['mode'],'complete=',a.get('complete'),
                  'format_fallback=',a.get('fallback_used'),
                  'finish_reason=',a.get('finish_reason'))
            excluded_ids.add(str(a['key']))
    valid_ids = [k for k in ids if k not in excluded_ids]
    print('All prompts:',len(ids),'| Clean paired subset:',len(valid_ids),'| Flagged:',len(excluded_ids))
    modes = ['direct','cot_final','cot_full']
    result = {}
    for rule in ['strict','loose']:
        scored = {}
        for mode in modes:
            path = root/mode/f'eval_results_{rule}.jsonl'
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            scored[mode] = {r['prompt']:r for r in rows}
            if len(scored[mode]) != len(ids):
                raise ValueError(f'{mode}/{rule}: number of unique scored prompts != number of inputs')
        result[rule] = {
            'all':report_difference(prompts,scored['direct'],scored['cot_final'], f'{rule.upper()} / all 60'),
            'clean_subset':report_difference([ids[k] for k in valid_ids],scored['direct'],scored['cot_final'],f'{rule.upper()} / excluding all audit flags')
        }
        f,raw=scored['cot_final'],scored['cot_full']
        a=sum(sum(r['follow_instruction_list']) for r in f.values())
        b=sum(sum(r['follow_instruction_list']) for r in raw.values())
        n=sum(len(r['follow_instruction_list']) for r in f.values())
        print(f'CoT Final -> Full ({rule}): {a}/{n} -> {b}/{n}; delta {100*(b-a)/n:+.2f} pp')
        result[rule]['full_vs_final']={'final_correct':a,'full_correct':b,'n_constraints':n,
                                      'delta_pp':round(100*(b-a)/n,3)}
    out=root/'paired_analysis_summary.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding='utf-8')
    print('\nSaved analysis:',out)
    print('No model calls made. Original generations and official evaluation files unchanged.')

if __name__=='__main__':
    main()
