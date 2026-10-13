"""Learn token spelling aliases from fitting entities only, never held-out labels."""
import argparse,collections,csv,zlib,re,functools
from pathlib import Path
from rapidfuzz.fuzz import ratio
ROOT=Path(__file__).resolve().parents[3]
@functools.lru_cache(maxsize=500000)
def phon(s):
    s=re.sub('[aeiouy]','',s).translate(str.maketrans({'v':'w','q':'k','z':'s'}))
    return re.sub(r'(.)\1+',r'\1',s)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fitting-buckets',type=int,default=100);ap.add_argument('--expanded-fitting',action='store_true');ap.add_argument('--fitting-stop',type=int,default=400);ap.add_argument('--output',type=Path);ap.add_argument('--work',type=Path,default=ROOT/'work');args=ap.parse_args()
    if not 200<=args.fitting_stop<=1000:ap.error('--fitting-stop must be between 200 and 1000')
    work=args.work;owners={};fitids=set();refnames={};refaddresses={};counts=collections.Counter()
    data=Path((work/'data_path.txt').read_text().strip()) if (work/'data_path.txt').exists() else ROOT/'dataset/student_resource/dataset'
    gt=work/'sample_truth.tsv' if args.fitting_buckets==20 and not args.expanded_fitting else data/'train/train_ground_truth.tsv'
    with gt.open() as inp:
        for l in inp:
            sid,targets=l.rstrip('\n').split('\t');bucket=zlib.crc32(sid.encode())%1000
            fitting=bucket<args.fitting_buckets or (args.expanded_fitting and 200<=bucket<args.fitting_stop)
            if sid=='source1_entity_id' or not fitting or 20<=bucket<30:continue
            fitids.add(sid)
            for t in targets.split(',') if targets else []:owners[t]=sid
    with (work/'train_source1.norm.tsv').open() as f:
        for l in f:
            sid,c,n,a=l.rstrip('\n').split('\t')[:4]
            if sid in fitids:
                refnames[sid]=list(set(n.split()));refaddresses[sid]=list(set(a.split()));counts.update(set(n.split()));counts.update('@'+t for t in set(a.split()))
    observed=collections.defaultdict(collections.Counter);total=collections.Counter()
    for source in [2,3]:
        with (work/f'train_source{source}.norm.tsv').open() as f:
            for l in f:
                sid,c,n,a=l.rstrip('\n').split('\t')[:4]
                if sid not in owners:continue
                canonical=refnames[owners[sid]]
                for t in set(n.split()):
                    if len(t)<3:continue
                    total[t]+=1
                    if t in canonical:observed[t][t]+=1;continue
                    ranked=sorted(((max(ratio(t,u)/100,.90*ratio(phon(t),phon(u))/100),u) for u in canonical if len(u)>=3),reverse=True)
                    if ranked and ranked[0][0]>=.58 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>.06):observed[t][ranked[0][1]]+=1
                canonical=refaddresses[owners[sid]]
                for t in set(a.split()):
                    if len(t)<3 or t.isdigit():continue
                    key='@'+t;total[key]+=1
                    if t in canonical:observed[key][key]+=1;continue
                    ranked=sorted(((max(ratio(t,u)/100,.90*ratio(phon(t),phon(u))/100),u) for u in canonical if len(u)>=3 and not u.isdigit()),reverse=True)
                    if ranked and ranked[0][0]>=.65 and (len(ranked)==1 or ranked[0][0]-ranked[1][0]>.08):observed[key]['@'+ranked[0][1]]+=1
    aliases={}
    for t,choices in observed.items():
        u,k=choices.most_common(1)[0]
        if t!=u and k>=3 and k/total[t]>=.80 and counts[u]>counts[t]*2:aliases[t]=u
    # Transliteration of legal forms carries no business identity information.
    legal_phon={phon(t) for t in ['private','pvt','limited','ltd','llp','llc','corporation','incorporated']}
    for t in total:
        if t.startswith('@'):continue
        if phon(t) in legal_phon:aliases[t]='-'
    with (args.output or work/'aliases.tsv').open('w') as f:
        for t,u in sorted(aliases.items()):f.write(t+'\t'+u.lstrip('@')+'\n')
    print('learned',len(aliases),'aliases from',len(fitids),'fitting entities',flush=True)
if __name__=='__main__':main()
