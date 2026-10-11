"""Probe pinned public history for actual missing bytes; never create substitutes."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = '6b12baa67e72a544a1c445c546ebc73da1ddfdb4'
MAIN = 'cf22ddd8e46ee83ddd9565b058e1df4e362377ea'
RESULT = ROOT/'docs/research-handoff/receipts/recovery-full-suite-20261011/repository-test-results.json'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).splitlines()


def calculate():
    results = json.loads(RESULT.read_text())
    groups = {}
    for case in results['items']:
        if case['status'] != 'failed':
            continue
        msg = case['message']
        if 'No such file or directory:' in msg:
            path = re.search(r"No such file or directory: '([^']+)'",msg).group(1)
        elif msg.startswith('FileNotFoundError: skill_missing:'):
            path = msg.split(':',3)[-1]
        elif msg == 'ValueError: scope_path_required_without_fixture':
            path = '.omx/goals/autoresearch/execute-evidencegate-engineering-paper-plan-v2-f/artifacts/g0/g0-execution-scope-v2.json'
        else:
            raise ValueError('Unclassified failure:' + msg)
        groups.setdefault(path,[]).append(case['module']+'::'+case['test'])
    commits = git('rev-list',SOURCE,MAIN)
    trees = {commit:set(git('ls-tree','-r','--name-only',commit)) for commit in commits}
    all_paths = set().union(*trees.values())
    rows = []
    for path,tests in sorted(groups.items()):
        if path.startswith('research_tools/remote/'):
            category = 'unpublished_remote_launcher'
        elif '/skills/' in path:
            category = 'unpublished_skill_body'
        elif path.startswith('docs/'):
            category = 'unpublished_protocol_or_claims_document'
        else:
            category = 'unpublished_historical_JSON_or_manifest'
        rows.append({'missing_path':path,'blocked_tests':tests,'blocked_test_count':len(tests),
                     'category':category,'commits_containing_exact_path':[c for c in commits if path in trees[c]],
                     'same_basename_candidates':sorted(p for p in all_paths if Path(p).name==Path(path).name),
                     'restoration_status':'no_committed_bytes_found_no_substitute_created'})
    secondary = ['research_tools/execution_policy/skills/count_schedule/SKILL.md',
                 'research_tools/remote/build_codec_derivatives.py',
                 'research_tools/remote/build_development_video_receipt.py']
    dependencies = [{'missing_path':p,'commits_containing_exact_path':[c for c in commits if p in trees[c]],
                     'same_basename_candidates':sorted(q for q in all_paths if Path(q).name==Path(p).name),
                     'role':'unexecuted_import_or_next_required_skill_not_reached'} for p in secondary]
    categories=Counter()
    for r in rows:
        categories[r['category']]+=r['blocked_test_count']
    return {'source_commit':SOURCE,'main_commit':MAIN,'source_results_sha256':hashlib.sha256(RESULT.read_bytes()).hexdigest(),
            'branches_queried':['main','task/research-handoff-v1'],'tags_found':[],
            'complete_history_not_shallow':True,'commits_examined':commits,'commit_count':len(commits),
            'failed_test_count':sum(len(v) for v in groups.values()),'primary_missing_path_count':len(rows),
            'failure_categories':dict(categories),'missing_path_groups':rows,
            'collection_errors':results['unexecuted_due_to_collection'],
            'secondary_code_dependencies':dependencies,
            'version_intent_evidence':{'publication_root_commit':MAIN,
                                     'publication_checks':'docs/publication-checks.md',
                                     'source_manifest':'SOURCE_MANIFEST.json',
                                     'source_history_included':False,
                                     'published_snapshot_excludes_remote_launchers_and_original_research_evidence':True},
            'limits':['First failing paths only; resolving a root may reveal more transitive files',
                      'No original repository history, private artifacts, GT or licenses inferred',
                      'Same basename is not same version or same evidence; no alternate is substituted'],
            'restored_historical_files':0,'assertions_modified':False,'gates_modified':False}


if __name__=='__main__':
    result=calculate()
    path=HERE/'history_probe_results.json'
    if sys.argv[1:]==['--generate']:
        path.write_text(json.dumps(result,indent=2)+'\n')
    elif sys.argv[1:]:
        raise SystemExit('Only optional --generate is supported')
    elif result!=json.loads(path.read_text()):
        raise SystemExit('History probe results differ')
    print(json.dumps({k:result[k] for k in ('commit_count','failed_test_count','primary_missing_path_count','failure_categories','restored_historical_files')},indent=2))
