"""Match training target/reference density without using evaluation labels."""
import argparse,collections,json,zlib
from pathlib import Path

def prepare(work):
    counts={}
    for split in ['train','test']:
        counts[split]={}
        for source in [1,2,3]:
            c=collections.Counter()
            with (work/f'{split}_source{source}.norm.tsv').open() as f:
                for line in f:c[line.split('\t',2)[1]]+=1
            counts[split][source]=c
    # Preserve development and reserved audit references. A second independent
    # identifier hash samples the remaining reference population, without labels.
    protected_fraction=.11
    rates={}
    for country,n in counts['train'][1].items():
        train_ratio=(counts['train'][2][country]+counts['train'][3][country])/n
        test_ratio=(counts['test'][2][country]+counts['test'][3][country])/counts['test'][1][country]
        desired=min(1.,train_ratio/test_ratio)
        rates[country]=max(0.,min(1.,(desired-protected_fraction)/(1-protected_fraction)))
    kept=set();retained=collections.Counter();total=collections.Counter()
    output=work/'train_source1.dense.norm.tsv';partial=output.with_suffix('.partial')
    with (work/'train_source1.norm.tsv').open() as inp,partial.open('w') as out:
        for line in inp:
            sid,country=line.split('\t',2)[:2];group=zlib.crc32(sid.encode())%1000
            protected=20<=group<30 or 100<=group<200
            total[country]+=1
            if protected or zlib.crc32(('density:'+sid).encode())/2**32<rates[country]:
                out.write(line);kept.add(sid);retained[country]+=1
    partial.replace(output)
    truth=work/'sample_truth.tsv';temporary=work/'sample_truth.dense.partial'
    with truth.open() as inp,temporary.open('w') as out:
        for line in inp:
            if line.split('\t',1)[0] in kept:out.write(line)
    temporary.replace(truth)
    cfg=json.loads((work/'candidate_config.json').read_text())
    cfg.update(density_matched=True,reference_population_file=output.name,reference_population_counts=dict(retained),original_reference_population_counts=dict(total))
    (work/'candidate_config.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps({'retention_rates_outside_protected_groups':rates,'retained':retained,'original':total,'all_targets_retained':True,'evaluation_and_audit_references_preserved':True},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);a=ap.parse_args();prepare(a.work)
