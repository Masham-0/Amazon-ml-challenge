"""Evaluate frozen decisions on training businesses outside development buckets."""
import argparse,csv,json,zlib
from pathlib import Path
import numpy as np
from export import DTYPE
from train import truth_data,metric,exclusive
from decision import selected
ROOT=Path(__file__).resolve().parents[3]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--report',type=Path,default=ROOT/'reports/final_audit.json');ap.add_argument('--audit-start',type=int,default=100);ap.add_argument('--audit-buckets',type=int,default=5);a=ap.parse_args();stop=a.audit_start+a.audit_buckets
    ids=[];countries=[]
    with (a.work/'test.refs.tsv').open() as inp:
        for line in inp:sid,c,_=line.rstrip('\n').split('\t');ids.append(sid);countries.append(c)
    ids=np.array(ids);countries=np.array(countries);groups=np.array([zlib.crc32(s.encode())%1000 for s in ids]);mask=(groups>=a.audit_start)&(groups<stop)
    data=Path((a.work/'data_path.txt').read_text().strip()) if (a.work/'data_path.txt').exists() else ROOT/'dataset/student_resource/dataset'
    with (data/'train/train_ground_truth.tsv').open() as inp,(a.work/'sample_truth.tsv').open('w') as out:
        for row in csv.DictReader(inp,delimiter='\t'):
            if a.audit_start<=zlib.crc32(row['source1_entity_id'].encode())%1000<stop:out.write(row['source1_entity_id']+'\t'+row['matched_entity_ids']+'\n')
    truth,pairs,sizes=truth_data(a.work,ids);rows=np.memmap(a.work/'test.scores.bin',dtype=DTYPE,mode='r');packed=(rows['ref'].astype(np.uint64)<<32)|rows['target'];y=np.isin(packed,pairs);cfg=json.loads((a.work/'config.json').read_text());p=selected(rows,rows['p'],cfg).astype(np.float32);threshold=.5
    exclusion='Reference labels excluded from dictionaries, both fitting stages, and threshold selection.'
    if cfg.get('strict_audit_targets'):exclusion+=' Their target records were also excluded from supervised fitting as negative examples.'
    else:exclusion+=' Target records may have occurred as negative examples during fitting, so this is not a strictly record-disjoint audit.'
    measured=metric(rows,p,y,sizes,mask,threshold)
    report={'protocol':f'CRC32 buckets {a.audit_start}–{stop-1}: '+exclusion+' Full-population candidate generation and complete scoring context for audit references and their competing references.','strict_audit_targets':bool(cfg.get('strict_audit_targets')),'threshold_frozen':threshold,'heldout':measured[0],'countries':{}}
    report['threshold_frozen']=cfg['threshold']
    if cfg.get('adaptive_thresholds'):report['adaptive_thresholds']=cfg['adaptive_thresholds']
    for c in sorted(set(countries)):report['countries'][c]=metric(rows,p,y,sizes,mask&(countries==c),threshold)[0]
    ceiling=np.bincount(rows['ref'][y],minlength=len(ids));upper=np.divide(1.25*ceiling,ceiling+.25*sizes,out=np.ones(len(sizes)),where=sizes>0)
    report['blocking']={'pair_recall':float(ceiling[mask].sum()/sizes[mask].sum()),'macro_f05_ceiling':float(upper[mask].mean()),'candidate_pairs_per_reference':float(np.bincount(rows['ref'],minlength=len(ids))[mask].mean())}
    a.report.parent.mkdir(exist_ok=True,parents=True)
    entity_path=a.report.with_suffix('.entities.tsv')
    with entity_path.open('w') as out:
        out.write('entity_id\tcountry\tf05\tpredicted_matches\ttrue_matches\ttrue_positive_matches\n')
        for i in np.flatnonzero(mask):out.write(f'{ids[i]}\t{countries[i]}\t{measured[1][i]:.17g}\t{measured[2][i]}\t{sizes[i]}\t{measured[3][i]}\n')
    report['entity_scores_file']=entity_path.name
    a.report.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
