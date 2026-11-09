"""Optional supplied test truth: final reporting only, never imported by training workers."""
from pathlib import Path
import json,zipfile
from lsi_data import records,read_json,atomic_json,target_matrix,metric

def evaluate_supplied_test(root,submission,output=None):
    raw=read_json(Path(root)/'Test/test.json')
    try:truth=records(raw,'dev')
    except ValueError as error:
        if 'labels must be a list' in str(error):return {'available':False,'note':'No valid supplied test truth. No test score was computed.'}
        raise
    with zipfile.ZipFile(submission) as archive:pred=json.loads(archive.read('prediction.json'))
    byid={r['id']:r['pred'] for r in pred}
    if len(byid)!=len(pred) or set(byid)!={r['id'] for r in truth}:raise ValueError('Test truth/prediction IDs differ.')
    dimension=max((j for r in truth for j in r['gold']),default=-1)+1
    if dimension<1:raise ValueError('Supplied test truth has no labels.')
    if any(type(j) is not int or j<0 or j>=dimension for row in byid.values() for j in row):raise ValueError('Prediction label outside supplied evaluator truth dimension.')
    y=target_matrix(truth,dimension);p=target_matrix([{'gold':byid[r['id']]} for r in truth],dimension)
    scores={'available':True,**metric(y,p),'evaluation_only':True,'submission':str(submission),
        'note':'Frozen predictions scored after all development-based choices. No test-driven model, threshold, fusion or submission selection.'}
    if output:atomic_json(output,scores)
    return scores
