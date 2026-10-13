"""Export the exact scored candidate set and exclusive final matches."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from decision import selected as select_matches
ROOT=Path(__file__).resolve().parents[3]
DTYPE=np.dtype([('ref','<u4'),('target','<u4'),('p','<f4')])
def target_id(t):return ('S3-' if t>>31 else 'S2-')+str(t&0x7fffffff)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work');ap.add_argument('--output',type=Path,default=ROOT/'output');ap.add_argument('--report',type=Path,default=ROOT/'reports/submission_report.json');ap.add_argument('--threshold',type=float);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    config=json.loads((a.work/'config.json').read_text());threshold=a.threshold if a.threshold is not None else config['threshold']
    refs=[l.split('\t')[0] for l in (a.work/'test.refs.tsv').read_text().splitlines()];rows=np.memmap(a.work/'test.scores.bin',dtype=DTYPE,mode='r');print('pairs',len(rows),'references',len(refs),'threshold',threshold,flush=True)
    # Target ownership is exclusive; optional reference-level cutoffs are
    # calibrated on disjoint tuning entities before test inference.
    config['threshold']=threshold
    if a.threshold is not None:config.pop('adaptive_thresholds',None)
    selected=select_matches(rows,rows['p'],config)
    order=np.argsort(rows['ref'],kind='stable');counts=np.bincount(rows['ref'],minlength=len(refs));offsets=np.r_[0,np.cumsum(counts)]
    nmatch=0;singletons=0;country_counts={}
    with (a.output/'matching_results.tsv').open('w') as match,(a.output/'candidate_pairs.tsv').open('w') as cand:
        match.write('source1_entity_id\tmatched_entity_ids\n');cand.write('source1_entity_id\tcandidate_entity_ids\n')
        for i,sid in enumerate(refs):
            ix=order[offsets[i]:offsets[i+1]];targets=[target_id(int(t)) for t in rows['target'][ix]];matches=[target_id(int(t)) for t in rows['target'][ix[selected[ix]]]]
            # No post-model pruning of the reported candidate list is allowed.
            if len(targets)!=len(set(targets)):raise ValueError('duplicate candidate')
            match.write(sid+'\t'+','.join(matches)+'\n');cand.write(sid+'\t'+','.join(targets)+'\n');nmatch+=len(matches);singletons+=not matches
            if i%500000==0:print('exported',i,flush=True)
    report={'test_reference_entities':len(refs),'scored_candidate_pairs':int(len(rows)),'mean_candidates_per_reference':float(counts.mean()),'median_candidates_per_reference':float(np.median(counts)),'p95_candidates_per_reference':float(np.quantile(counts,.95)),'final_matches':nmatch,'predicted_singletons':singletons,'threshold':threshold}
    if config.get('adaptive_thresholds'):report['adaptive_thresholds']=config['adaptive_thresholds']
    report['model_hashes']={name:hashlib.sha256((a.work/name).read_bytes()).hexdigest() for name in ['config.json','aliases.tsv']+config.get('ensemble',['model.txt'])+(['collective_model.txt'] if config.get('collective') else [])}
    a.report.parent.mkdir(exist_ok=True,parents=True);a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
