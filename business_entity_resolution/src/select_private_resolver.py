"""Honor an explicit local request before the first full-test model scoring.

Candidate retrieval and frozen features stay identical. Completed private trials
may replace only the final classifier and its measured configuration. This
avoids rescoring the full test set twice when cloud training finishes in time.
The request is absent from normal organizer reproduction and from the archive.
"""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path

COMPATIBLE=['ensemble','features','second_stage_features','gate','gap','candidate_limit','name_only_candidate_limit','name_only_gap','tfidf','copies','wide_blocking']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def compatible(first,second):
    old=json.loads((first/'config.json').read_text());new=json.loads((second/'config.json').read_text())
    for key in COMPATIBLE:assert old.get(key)==new.get(key),key
    for name in old['ensemble']+['aliases.tsv']:assert sha(first/name)==sha(second/name),name
    for sid in [1,2,3]:assert (first/f'test_source{sid}.norm.tsv').samefile(second/f'test_source{sid}.norm.tsv')
    return new
def best_variant(scores):
    selected='v9';best=scores[selected]
    if scores.get('v10weighted',-1)>best+.0005:selected='v10weighted';best=scores[selected]
    if scores.get('v11blend',-1)>best+.0003 and scores['v11blend']>scores['v9']+.0005:selected='v11blend'
    return selected
def atomic_copy(source,target):
    temporary=target.with_name(target.name+'.selection.partial')
    shutil.copy2(source,temporary);temporary.replace(target)

def choose_before_scoring(work):
    work=Path(work).resolve();request=work/'private_selection_request.json'
    if not request.exists():return
    spec=json.loads(request.read_text());root=Path(spec['project_root']).resolve()
    assert work==root/'work/v9'
    reports=root/'reports/v9';record=reports/'selected_private_resolver.json'
    if record.exists():return
    assert sha(work/'collective_model.txt')==spec['baseline_model_sha256']
    baseline=json.loads((reports/'context_decision_validation_report.json').read_text())['active_tuning']['macro_f05']
    result={'selection_uses':'CRC100–102 complete-context tuning only','test_labels_used':False,'candidate_retrieval_changed':False,'selected_private_variant':'v9','baseline_model_sha256':spec['baseline_model_sha256'],'tuning_macro_f05':{'v9':baseline}}
    state=root/'work/private_premeasure_status.json'
    if not state.exists() or not json.loads(state.read_text())['stage'].startswith('Private models measured;'):
        result['note']='Private comparisons were not complete before first scoring; serial late comparison remains available.'
        record.write_text(json.dumps(result,indent=2));print(result['note'],flush=True);return
    for version in ['v10weighted','v11blend']:
        result['tuning_macro_f05'][version]=json.loads((root/'reports'/version/'context_decision_validation_report.json').read_text())['active_tuning']['macro_f05']
    chosen=best_variant(result['tuning_macro_f05'])
    if chosen=='v9':
        result['note']='Private comparisons did not clear the required tuning gain.'
        record.write_text(json.dumps(result,indent=2));print(result['note'],flush=True);return
    candidate=root/'work'/chosen;candidate_reports=root/'reports'/chosen
    new=compatible(work,candidate)
    assert new.get('context_decision_calibration')
    # Preserve the measured baseline before replacing any final-stage artifact.
    snapshot=root/'work/v9_before_private_selection';snapshot.mkdir(exist_ok=False)
    old_reports=snapshot/'reports';shutil.copytree(reports,old_reports)
    names=['collective_model.txt','config.json']+new.get('blend_component_models',[])
    if (candidate/'collective_validation_model.txt').exists():names.append('collective_validation_model.txt')
    absent=[]
    for name in names:
        if (work/name).exists():shutil.copy2(work/name,snapshot/name)
        else:absent.append(name)
    try:
        for name in names:atomic_copy(candidate/name,work/name)
        for name in ['validation_report.json','context_decision_validation_report.json','context_calibration_initial_audit.json','remote_training_report.json','blend_report.json']:
            if (candidate_reports/name).exists():atomic_copy(candidate_reports/name,reports/name)
        source=Path(__file__).resolve().parent
        for script,report in [('verify.py','runtime_verification.json'),('verify_context.py','context_verification.json')]:
            subprocess.run([sys.executable,str(source/script),'--work',str(work),'--fixture',str(root/'work/equivalence'),'--report',str(reports/report)],check=True)
        result.update(selected_private_variant=chosen,selected_model_sha256=sha(work/'collective_model.txt'),native_and_context_checks_passed=True,baseline_snapshot=str(snapshot))
    except BaseException as error:
        for name in names:
            if name in absent:(work/name).unlink(missing_ok=True)
            else:atomic_copy(snapshot/name,work/name)
        for path in reports.iterdir():
            if path.is_file() and not (old_reports/path.name).exists():path.unlink()
        for path in old_reports.iterdir():
            if path.is_file():atomic_copy(path,reports/path.name)
        result['optional_private_selection_error']=type(error).__name__+': '+str(error)
        result['note']='Baseline restored before any full-test final-stage predictions.'
    record.write_text(json.dumps(result,indent=2))
    selection=root/'reports/deadline_model_selection.json'
    original=json.loads(selection.read_text());original['final_stage_private_selection']=result
    temporary=selection.with_suffix('.partial');temporary.write_text(json.dumps(original,indent=2));temporary.replace(selection)
    shutil.copy2(selection,reports/'deadline_model_selection.json')
    print('Frozen classifier selected before full-test scoring:',result['selected_private_variant'],flush=True)
