"""Run unchanged organizer validation functions with bounded output batches.

Global reference coverage/duplicates and every target ID are checked. Only the
large output dictionaries are partitioned; no rows or candidate IDs are skipped.
"""
import argparse,gc,hashlib,importlib.util,itertools,json,resource,tempfile
from pathlib import Path


def validate(organizer,matching,candidate,test_dir,report,batch_size=75000):
    # Fail cleanly rather than exhausting desktop memory if inputs grow.
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    spec=importlib.util.spec_from_file_location('organizer_validator',organizer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    remaining=module.read_ids(test_dir/'test_source1.tsv');required_count=len(remaining)
    warnings=[];valid=module.load_match_targets(test_dir,warnings)
    if warnings or valid is None:raise ValueError('Complete target ID checks unavailable: '+str(warnings))
    print('All source IDs loaded:',required_count,'references;',len(valid),'targets',flush=True)
    hashes=[hashlib.sha256(),hashlib.sha256()];count=batches=0
    with tempfile.TemporaryDirectory(prefix='shawarma-validation-') as directory:
        files=[Path(directory)/matching.name,Path(directory)/candidate.name]
        with matching.open() as mf,candidate.open() as cf:
            headers=[mf.readline(),cf.readline()]
            expected=[module.MATCHING_HEADER,module.CANDIDATE_HEADER]
            for header,columns,h in zip(headers,expected,hashes):
                if [x.strip().lower() for x in header.rstrip('\n').split('\t')]!=columns:raise ValueError('Invalid submission header')
                h.update(header.encode())
            iterator=itertools.zip_longest(mf,cf)
            while True:
                required=set()
                with files[0].open('w') as mo,files[1].open('w') as co:
                    mo.write(headers[0]);co.write(headers[1])
                    for first,second in itertools.islice(iterator,batch_size):
                        if first is None or second is None:raise ValueError('Output row counts differ')
                        sid=first.partition('\t')[0]
                        if second.partition('\t')[0]!=sid:raise ValueError('Output reference row order differs')
                        if sid not in remaining:raise ValueError('Unknown or duplicate reference ID: '+sid)
                        remaining.remove(sid);required.add(sid);count+=1
                        mo.write(first);co.write(second)
                        hashes[0].update(first.encode());hashes[1].update(second.encode())
                if not required:break
                errors=[]
                matches=module.validate_id_list_file(str(files[0]),module.MATCHING_HEADER,'matched_entity_ids',required,valid,errors)
                candidates=module.validate_id_list_file(str(files[1]),module.CANDIDATE_HEADER,'candidate_entity_ids',required,valid,errors)
                if errors or matches is None or candidates is None:raise ValueError('Organizer validation failed: '+str(errors))
                if any(ids-candidates[sid] for sid,ids in matches.items()):raise ValueError('Final match absent from candidates')
                del matches,candidates;gc.collect();batches+=1
                print('Validated',count,'/',required_count,'references with all target ID checks',flush=True)
    if remaining:raise ValueError(f'Missing {len(remaining)} required references')
    result={'passed':True,'mode':'Unchanged organizer per-file validation functions in bounded batches; global exact coverage and duplicates checked before partitioning','references':count,'batches':batches,'batch_size':batch_size,'all_target_ids_checked':True,'all_final_matches_in_candidates':True,'organizer_sha256':hashlib.sha256(organizer.read_bytes()).hexdigest(),'output_sha256':{matching.name:hashes[0].hexdigest(),candidate.name:hashes[1].hexdigest()},'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--organizer',type=Path,required=True);ap.add_argument('--matching',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--test-dir',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    validate(args.organizer,args.matching,args.candidate,args.test_dir,args.report)
