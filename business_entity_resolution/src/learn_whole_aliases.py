"""Learn recurring whole-name variants across distinct fitting businesses."""
import argparse,collections,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work');ap.add_argument('--expanded-fitting',action='store_true');ap.add_argument('--fitting-stop',type=int,default=400);a=ap.parse_args();work=a.work
    if not 200<=a.fitting_stop<=1000:ap.error('--fitting-stop must be between 200 and 1000')
    alias=dict(l.rstrip('\n').split('\t') for l in (work/'aliases.tsv').open() if not l.startswith('%'))
    def canonical(n):return ' '.join(sorted(set(alias.get(t,t) for t in n.split() if alias.get(t,t)!='-')))
    data=Path((work/'data_path.txt').read_text().strip()) if (work/'data_path.txt').exists() else ROOT/'dataset/student_resource/dataset'
    owners={};fit=set();names={}
    for l in (data/'train/train_ground_truth.tsv').open():
        sid,ts=l.rstrip('\n').split('\t');g=zlib.crc32(sid.encode())%1000
        fitting=g<100 or (a.expanded_fitting and 200<=g<a.fitting_stop)
        if sid=='source1_entity_id' or not fitting or 20<=g<30:continue
        fit.add(sid)
        for t in ts.split(',') if ts else []:owners[t]=sid
    for l in (work/'train_source1.norm.tsv').open():
        sid,_,n,*_=l.rstrip('\n').split('\t')
        if sid in fit:names[sid]=canonical(n)
    # A qualifying alias has at least 95% agreement, so it has a strict
    # majority. Find that majority with bounded state, then count its exact
    # support in a second pass. This is equivalent to retaining every vote.
    observed={}
    for source in [2,3]:
        for l in (work/f'train_source{source}.norm.tsv').open():
            sid,_,n,*_=l.rstrip('\n').split('\t')
            if sid not in owners:continue
            n=canonical(n);u=names[owners[sid]]
            if not n or not u:continue
            if n not in observed:observed[n]=[u,1,1]
            else:
                state=observed[n];state[2]+=1
                if state[1]==0:state[0]=u;state[1]=1
                elif state[0]==u:state[1]+=1
                else:state[1]-=1
    candidates={n:[u,total,0,set()] for n,(u,balance,total) in observed.items() if n!=u and total>=3}
    del observed
    for source in [2,3]:
        for l in (work/f'train_source{source}.norm.tsv').open():
            sid,_,n,*_=l.rstrip('\n').split('\t')
            if sid not in owners:continue
            n=canonical(n);state=candidates.get(n)
            if state is None or names[owners[sid]]!=state[0]:continue
            state[2]+=1
            if len(state[3])<3:state[3].add(owners[sid])
    full={}
    for n,(u,total,count,distinct) in candidates.items():
        if count/total>=.95 and len(distinct)>=3:full['%'+n]=u
    with (work/'aliases.tsv').open('w') as out:
        for k,v in sorted({**alias,**full}.items()):out.write(k+'\t'+v+'\n')
    print('whole-name mappings',len(full),'distinct fitting references',len(fit),flush=True)
if __name__=='__main__':main()
