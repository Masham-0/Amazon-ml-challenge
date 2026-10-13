"""Normalize supplied records; no external data or entity lookup is used."""
import argparse, csv, json, re, time, zlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[3]
WORDS = re.compile(r"[a-z0-9]+")
URL = re.compile(r"(?:https?://)?(?:www\.)?[a-z0-9-]+\.(?:com|in|org|net|co|fr)(?:/\S*)?", re.I)
LEGAL = set("inc incorporated corporation corp llc llp ltd limited private pvt public company co plc pc sa sas sasu sarl eurl sci association societé societe m s ms the and de du des et la le les".split())
ALIASES = dict(street='st', road='rd', avenue='ave', drive='dr', boulevard='blvd', court='ct', lane='ln', place='pl', highway='hwy', apartment='apt', apartments='apt', suite='ste', north='n', south='s', east='e', west='w', floor='fl', number='no', route='rte', building='bldg', private='pvt', limited='ltd', corporation='corp', incorporated='inc', allée='allee', av='ave', bd='blvd', bl='blvd', r='rue')
STATES = {'new york':'ny','north carolina':'nc','south carolina':'sc','new jersey':'nj','new mexico':'nm','new hampshire':'nh','rhode island':'ri','west virginia':'wv','north dakota':'nd','south dakota':'sd','district of columbia':'dc','texas':'tx','california':'ca','florida':'fl','pennsylvania':'pa','ohio':'oh','illinois':'il','georgia':'ga','washington':'wa','virginia':'va','arizona':'az','massachusetts':'ma','tennessee':'tn','indiana':'in','missouri':'mo','maryland':'md','wisconsin':'wi','colorado':'co','minnesota':'mn','alabama':'al','louisiana':'la','kentucky':'ky','oregon':'or','oklahoma':'ok','connecticut':'ct','iowa':'ia','mississippi':'ms','arkansas':'ar','utah':'ut','kansas':'ks','nevada':'nv','nebraska':'ne','idaho':'id','hawaii':'hi','maine':'me','montana':'mt','delaware':'de','alaska':'ak','vermont':'vt','wyoming':'wy'}
STATE_RE = re.compile(r'\b('+'|'.join(sorted(STATES,key=len,reverse=True))+r')\b')

def clean(text, name=False):
    text = unidecode(text).lower()
    text = re.sub(r'\b(?:s\.?\s*a\.?\s*r\.?\s*l\.?|s\.?\s*a\.?\s*s\.?\s*u\.?|s\.?\s*a\.?\s*s\.?|s\.?\s*c\.?\s*i\.?)\b', ' ', text) if name else text
    if name:
        text = re.sub(r'\b(?:phone|tel|contact)\s*[:\-]?\s*[+()0-9 .-]{8,}', ' ', text)
        if URL.search(text):
            rest=URL.sub(' ',text)
            text=rest if len(WORDS.findall(rest)) else re.sub(r'\.(?:com|in|org|net|co|fr)\b',' ',text.replace('www.',''))
        text=text.replace('&',' and ')
    else:
        text=STATE_RE.sub(lambda m:STATES[m.group()],text)
    tokens=[ALIASES.get(t,t) for t in WORDS.findall(text)]
    if name: tokens=[t for t in tokens if t not in LEGAL]
    return ' '.join(tokens)[:500]

def normalize_file(args):
    source, dest, force = args
    if dest.exists() and not force: return str(dest)+' exists'
    started=time.time(); n=0
    with source.open() as inp, dest.with_suffix('.partial').open('w') as out:
        for row in csv.DictReader(inp,delimiter='\t'):
            rawname=' '.join(WORDS.findall(unidecode(row['business_name']).lower().replace('&',' and ')))[:500]
            out.write('\t'.join([row['entity_id'],row['country'],clean(row['business_name'],True),clean(row['business_address']),rawname])+'\n')
            n+=1
            if n%1000000==0: print(source.name,n,round(time.time()-started,1),flush=True)
    dest.with_suffix('.partial').replace(dest)
    return f'{source.name}: {n} rows, {time.time()-started:.1f}s'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'dataset/student_resource/dataset');ap.add_argument('--work',type=Path,default=ROOT/'work');ap.add_argument('--split',choices=['train','test','both'],default='both');ap.add_argument('--workers',type=int,default=3);ap.add_argument('--force',action='store_true');ap.add_argument('--sample-buckets',type=int,default=100);ap.add_argument('--expanded-sample',action='store_true');a=ap.parse_args();a.work.mkdir(exist_ok=True,parents=True)
    (a.work/'data_path.txt').write_text(str(a.data.resolve())+'\n')
    jobs=[(a.data/split/f'{split}_source{s}.tsv',a.work/f'{split}_source{s}.norm.tsv',a.force) for split in (['train','test'] if a.split=='both' else [a.split]) for s in [1,2,3]]
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(normalize_file,jobs):print(r,flush=True)
    if a.split!='test':
        with (a.data/'train/train_ground_truth.tsv').open() as inp,(a.work/'sample_truth.tsv').open('w') as out:
            for row in csv.DictReader(inp,delimiter='\t'):
                group=zlib.crc32(row['source1_entity_id'].encode())%1000
                if group<a.sample_buckets or (a.expanded_sample and 200<=group<400):
                    out.write(row['source1_entity_id']+'\t'+row['matched_entity_ids']+'\n')

if __name__=='__main__':main()
