"""Read supplied test truth only after predictions are frozen. No training consumer imports this."""
from pathlib import Path
import json,zipfile
from cj_data import records,read_json,atomic_json,metric

def evaluate_test_labels(root,submission,output=None):
    # Using the labeled-record parser here validates labels without putting them into model data.
    truth=records(read_json(Path(root)/'Test/test.json'),'dev')
    submission=Path(submission)
    if submission.suffix=='.zip':
        with zipfile.ZipFile(submission) as archive:predictions=json.loads(archive.read('prediction.json'))
    else:predictions=read_json(submission)
    if not isinstance(predictions,list) or not predictions:raise ValueError('Predictions must be a nonempty list.')
    pred={}
    for row in predictions:
        did=row.get('id');value=row.get('pred')
        if not isinstance(did,str) or not did or did in pred or type(value) is not int or value not in (0,1):
            raise ValueError('Invalid or repeated prediction ID/label.')
        pred[did]=value
    labels={}
    for row in truth:labels.setdefault(row['source_id'],set()).add(row['gold'])
    if set(pred)!=set(labels):raise ValueError('Test truth and prediction source-case IDs differ.')
    conflicts=[did for did,values in labels.items() if len(values)>1]
    record_scores=metric([row['gold'] for row in truth],[pred[row['source_id']] for row in truth])
    case_scores=None
    if not conflicts:case_scores=metric([next(iter(values)) for values in labels.values()],[pred[did] for did in labels])
    result={'evaluation_only':True,'submission':str(submission),'case_count':len(labels),'distinct_labeled_records':len(truth),
        'case_metrics':case_scores,'distinct_record_diagnostic':record_scores,
        'cases_with_conflicting_labels':len(conflicts),'conflicting_case_id_examples':conflicts[:20],
        'note':'No test-based checkpoint, threshold, fusion, ensemble or submission selection. Case metrics require unambiguous truth per original ID.'}
    if output is not None:atomic_json(output,result)
    return result
