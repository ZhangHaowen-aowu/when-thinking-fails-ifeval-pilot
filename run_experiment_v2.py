#!/usr/bin/env python3
"""Small-scale IFEval replication and evaluation-boundary audit (revised data-integrity version).

Runs a selected set of official IFEval examples through Direct vs explicit CoT prompts
on the same model and exports three compatible files for Google's official evaluator.
Requires: pip install openai (and install the upstream IFEval requirements separately).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

COT_PROMPT = """You will be given an instruction. You need to first carefully think step by step about the constraints and conditions in the instruction and then provide the answer. Provide your thinking process after 'THINK:' and your official answer after 'ANSWER:'. The given instruction is the following:
INSTRUCTION:
{question}
Now, generate your thinking process and final answer for the given instruction."""


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                try: yield json.loads(line)
                except Exception as ex: raise ValueError(f'{path}:{lineno}: {ex}') from ex


def write_jsonl(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for x in data: f.write(json.dumps(x, ensure_ascii=False)+'\n')


def key_of(record):
    return str(record.get('key', record['prompt']))


def check_unique(rows, label):
    keys=[key_of(r) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError(f'duplicate prompt/key in {label}')


def prepare(args):
    source=list(load_jsonl(args.source))
    if not 1 <= args.n <= len(source): raise ValueError('n out of range')
    check_unique(source, args.source)
    random.Random(args.seed).shuffle(source)
    # Only need official input items; no hardcoded dataset content or labels.
    chosen=source[:args.n]
    if any(not all(k in x for k in ['prompt','instruction_id_list','kwargs']) for x in chosen):
        raise ValueError('Source must be official IFEval input_data.jsonl with prompt/instruction_id_list/kwargs')
    write_jsonl(args.out, chosen)
    types={t for x in chosen for t in x['instruction_id_list']}
    print(f'Prepared {len(chosen)} prompts ({len(types)} instruction labels); seed={args.seed}. Saved {args.out}')


def parse_explicit_cot(raw):
    """Match the original authors' missing-marker fallback.

    Missing one/both explicit labels is a format failure, not necessarily a
    missing answer: treat the *entire* generation as the answer and record it
    for an explicit audit. Never infer or invent a hidden thinking trace.
    """
    a=raw.find('THINK:')
    b=raw.find('ANSWER:',a+6) if a>=0 else -1
    if a<0 or b<0:
        return {'parse_ok':bool(raw.strip()),'thinking':'','answer':raw.strip(),
                'parse_error':'' if raw.strip() else 'Empty generated response',
                'parse_warning':'Missing THINK:/ANSWER: markers; entire raw output treated as final answer (authors fallback)',
                'format_ok':False}
    thinking=raw[a+len('THINK:'):b].strip()
    answer=raw[b+len('ANSWER:'):].strip()
    if not answer:
        return {'parse_ok':False,'thinking':thinking,'answer':'',
                'parse_error':'Empty answer after ANSWER:','parse_warning':'','format_ok':False}
    return {'parse_ok':True,'thinking':thinking,'answer':answer,'parse_error':'',
            'parse_warning':'','format_ok':True}


def normalized_item(item):
    """Re-parse saved records consistently, without altering the raw file."""
    r=dict(item)
    if not r.get('ok') or r.get('finish_reason')=='length':
        return r
    if r.get('mode')=='cot':
        r.update(parse_explicit_cot(r.get('raw') or ''))
    elif r.get('mode')=='direct':
        raw=r.get('raw') or ''
        r.update({'parse_ok':bool(raw.strip()),'thinking':'','answer':raw.strip(),
                  'parse_error':'' if raw.strip() else 'Empty generated response'})
    return r


def valid_item(item):
    return bool(item.get('ok') and item.get('finish_reason')!='length' and
                item.get('parse_ok') and (item.get('answer') or '').strip())


def generate(args):
    from openai import OpenAI
    key=os.environ.get('OPENAI_API_KEY')
    model=args.model or os.environ.get('OPENAI_MODEL')
    base=args.base_url or os.environ.get('OPENAI_BASE_URL')
    if not key: raise RuntimeError('Set OPENAI_API_KEY in your environment; never include it in the report')
    if not model: raise RuntimeError('Supply --model MODEL or set OPENAI_MODEL')
    client=OpenAI(api_key=key,base_url=base) if base else OpenAI(api_key=key)
    rows=list(load_jsonl(args.samples))
    check_unique(rows,args.samples)
    if args.limit: rows=rows[:args.limit]
    out=Path(args.out);out.parent.mkdir(parents=True, exist_ok=True)
    existing=list(load_jsonl(out)) if out.exists() else []
    incompatible=[r for r in existing if r.get('ok') and r.get('model') != model]
    if incompatible: raise ValueError('Existing generations were made with a different model; use a new output file to avoid mixing runs')
    done={(str(r['key']),r['mode']) for old in existing for r in [normalized_item(old)] if valid_item(r)}
    modes=['direct','cot'] if args.mode=='both' else [args.mode]
    for idx,row in enumerate(rows,1):
        for mode in modes:
            k=key_of(row)
            if (k,mode) in done: continue
            prompt=row['prompt'] if mode=='direct' else COT_PROMPT.format(question=row['prompt'])
            last_error=None
            for trial in range(3):
                try:
                    answer=client.chat.completions.create(
                        model=model,
                        messages=[{'role':'user','content':prompt}],
                        temperature=0,
                        max_tokens=args.max_tokens)
                    msg=answer.choices[0].message
                    raw=msg.content or ''
                    if not raw: raise ValueError('No visible message.content; this model/API may put reasoning in a separate channel')
                    item={'key':k,'mode':mode,'model':model,'ok':True,'raw':raw,'prompt':row['prompt'],
                          'finish_reason':answer.choices[0].finish_reason,
                          'usage':answer.usage.model_dump() if answer.usage else None}
                    if mode=='cot': item.update(parse_explicit_cot(raw))
                    else: item.update({'parse_ok':True,'thinking':'','answer':raw.strip(),'parse_error':''})
                    if item['finish_reason']=='length':
                        item['parse_ok']=False;item['parse_error']='Truncated at max_tokens'
                    with out.open('a',encoding='utf-8') as f: f.write(json.dumps(item,ensure_ascii=False)+'\n')
                    if valid_item(item): done.add((k,mode))
                    print(f'{idx}/{len(rows)} {mode}: saved; parse_ok={item["parse_ok"]}')
                    break
                except Exception as ex:
                    last_error=str(ex)
                    print(f'{idx}/{len(rows)} {mode}: attempt {trial+1}/3 failed: {last_error}',file=sys.stderr)
                    if trial<2:time.sleep(2*(trial+1))
            else: print(f'FAIL {k} {mode}: {last_error}',file=sys.stderr)
    print(f'Done. Raw responses: {out}; resume the same command to skip saved runs.')


def export(args):
    """Select the first complete run, independently of accuracy.

    This never deletes original generations or silently drops benchmark items.
    All retries remain in raw JSONL for auditing. Truncated runs are ineligible.
    Missing-marker responses are handled exactly as in the paper authors' parser,
    with an explicit audit warning (not presented as a successful THINK parse).
    """
    rows=list(load_jsonl(args.samples))
    if args.limit:rows=rows[:args.limit]
    raw=list(load_jsonl(args.raw))
    by={}
    attempt_count={}
    for old in raw:
        r=normalized_item(old)
        pair=(str(r.get('key')),r.get('mode'))
        attempt_count[pair]=attempt_count.get(pair,0)+1
        if valid_item(r) and pair not in by:
            by[pair]=(r,attempt_count[pair])
    pairs=[];issues=[];audit=[]
    for row in rows:
        k=key_of(row)
        d_pair=by.get((k,'direct'))
        c_pair=by.get((k,'cot'))
        if not d_pair or not c_pair:
            missing=[m for m,rec in [('direct',d_pair),('cot',c_pair)] if not rec]
            issues.append(f'{k}: missing complete non-truncated run for {", ".join(missing)}')
            continue
        d,d_n=d_pair
        c,c_n=c_pair
        pairs.append((row,d,c))
        audit.extend([
            {'key':k,'mode':'direct','selected_attempt':d_n,'model':d.get('model'),
             'finish_reason':d.get('finish_reason'),'format_ok':True,'parse_warning':''},
            {'key':k,'mode':'cot','selected_attempt':c_n,'model':c.get('model'),
             'finish_reason':c.get('finish_reason'),
             'format_ok':c.get('format_ok',False),
             'parse_warning':c.get('parse_warning','')},
        ])
    if issues:
        print('\n'.join(issues[:15]),file=sys.stderr)
        raise ValueError(f'{len(issues)} missing complete pairs. Re-run only unfinished generations (no silent sample dropping).')
    root=Path(args.out_dir)
    write_jsonl(root/'input_data.jsonl',[x for x,d,c in pairs])
    for kind in ['direct','cot_final','cot_full']:
        directory=root/kind
        if kind=='direct':get=lambda d,c:d['answer']
        elif kind=='cot_final':get=lambda d,c:c['answer']
        else:get=lambda d,c:c['raw']
        write_jsonl(directory/'responses.jsonl',
                    [{'prompt':r['prompt'],'response':get(d,c)} for r,d,c in pairs])
    write_jsonl(root/'selected_runs_audit.jsonl',audit)
    warning_keys=[x['key'] for x in audit if x['mode']=='cot' and x['parse_warning']]
    print(f'Exported {len(pairs)} paired items to {root}; same official input subset in input_data.jsonl')
    print(f'CoT format-label fallback in {len(warning_keys)} cases: {warning_keys}')
    print('Selected first complete run per (key, mode); audit: selected_runs_audit.jsonl')
    print('Use the official IFEval evaluator separately on direct / cot_final / cot_full.')


def main():
    p=argparse.ArgumentParser(description='IFEval CoT replication with a final-vs-full-trace sensitivity check')
    sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('prepare'); x.add_argument('--source',required=True);x.add_argument('--n',type=int,default=60);x.add_argument('--seed',type=int,default=42);x.add_argument('--out',default='data/samples_60.jsonl');x.set_defaults(func=prepare)
    x=sub.add_parser('generate');x.add_argument('--samples',default='data/samples_60.jsonl');x.add_argument('--model');x.add_argument('--base-url');x.add_argument('--mode',choices=['direct','cot','both'],default='both');x.add_argument('--max-tokens',type=int,default=2048);x.add_argument('--limit',type=int);x.add_argument('--out',default='data/generations.jsonl');x.set_defaults(func=generate)
    x=sub.add_parser('export');x.add_argument('--samples',default='data/samples_60.jsonl');x.add_argument('--raw',default='data/generations.jsonl');x.add_argument('--limit',type=int);x.add_argument('--out-dir',default='data/eval_inputs');x.set_defaults(func=export)
    args=p.parse_args();args.func(args)
if __name__=='__main__':main()
