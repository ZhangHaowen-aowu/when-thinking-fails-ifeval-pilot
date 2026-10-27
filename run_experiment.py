#!/usr/bin/env python3
"""Small-scale IFEval replication and evaluation-boundary audit.

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
    # Matches authors' split logic, but unlike fallback logic, flags missing markers.
    a=raw.find('THINK:')
    b=raw.find('ANSWER:', a+6) if a>=0 else -1
    if a<0 or b<0: return {'parse_ok':False,'thinking':'','answer':'','parse_error':'Missing THINK:/ANSWER: markers'}
    thinking=raw[a+len('THINK:'):b].strip()
    answer=raw[b+len('ANSWER:'):].strip()
    if not answer: return {'parse_ok':False,'thinking':thinking,'answer':'','parse_error':'Empty answer after ANSWER:'}
    return {'parse_ok':True,'thinking':thinking,'answer':answer,'parse_error':''}


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
    done={(str(r['key']),r['mode']) for r in existing if r.get('ok') and r.get('parse_ok')}
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
                    done.add((k,mode))
                    print(f'{idx}/{len(rows)} {mode}: saved; parse_ok={item["parse_ok"]}')
                    break
                except Exception as ex:
                    last_error=str(ex)
                    print(f'{idx}/{len(rows)} {mode}: attempt {trial+1}/3 failed: {last_error}',file=sys.stderr)
                    if trial<2:time.sleep(2*(trial+1))
            else: print(f'FAIL {k} {mode}: {last_error}',file=sys.stderr)
    print(f'Done. Raw responses: {out}; resume the same command to skip saved runs.')


def export(args):
    rows=list(load_jsonl(args.samples))
    if args.limit:rows=rows[:args.limit]
    raw=list(load_jsonl(args.raw))
    by={}
    for item in raw:
        if item.get('ok'):by[(str(item['key']),item['mode'])]=item
    pairs=[];issues=[]
    for row in rows:
        k=key_of(row);d=by.get((k,'direct'));c=by.get((k,'cot'))
        if not d or not c:issues.append(f'{k}: missing direct or cot');continue
        if not d.get('parse_ok') or not c.get('parse_ok'):
            issues.append(f'{k}: invalid/truncated parse ({d.get("parse_error")}; {c.get("parse_error")})');continue
        pairs.append((row,d,c))
    if issues:
        print('\n'.join(issues[:12]),file=sys.stderr)
        raise ValueError(f'{len(issues)} missing/invalid pairs. Fix these before scoring (do not silently drop them).')
    root=Path(args.out_dir)
    write_jsonl(root/'input_data.jsonl',[x for x,d,c in pairs])
    for kind in ['direct','cot_final','cot_full']:
        directory=root/kind
        if kind=='direct':get=lambda d,c:d['answer']
        elif kind=='cot_final':get=lambda d,c:c['answer']
        else:get=lambda d,c:c['raw']
        write_jsonl(directory/'responses.jsonl', [{'prompt':r['prompt'],'response':get(d,c)} for r,d,c in pairs])
    print(f'Exported {len(pairs)} paired items to {root}; each mode has responses.jsonl; same subset in input_data.jsonl')
    print('Now run Google official IFEval evaluator in each of direct / cot_final / cot_full. See README.md.')


def main():
    p=argparse.ArgumentParser(description='IFEval CoT replication with a final-vs-full-trace sensitivity check')
    sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('prepare'); x.add_argument('--source',required=True);x.add_argument('--n',type=int,default=60);x.add_argument('--seed',type=int,default=42);x.add_argument('--out',default='data/samples_60.jsonl');x.set_defaults(func=prepare)
    x=sub.add_parser('generate');x.add_argument('--samples',default='data/samples_60.jsonl');x.add_argument('--model');x.add_argument('--base-url');x.add_argument('--mode',choices=['direct','cot','both'],default='both');x.add_argument('--max-tokens',type=int,default=2048);x.add_argument('--limit',type=int);x.add_argument('--out',default='data/generations.jsonl');x.set_defaults(func=generate)
    x=sub.add_parser('export');x.add_argument('--samples',default='data/samples_60.jsonl');x.add_argument('--raw',default='data/generations.jsonl');x.add_argument('--limit',type=int);x.add_argument('--out-dir',default='data/eval_inputs');x.set_defaults(func=export)
    args=p.parse_args();args.func(args)
if __name__=='__main__':main()
