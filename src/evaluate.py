"""Evaluate the frozen model on held-out test data, without any setting changes."""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score
from .common import ROOT,normalize,sha256,write_json

def metrics(y,pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    cm=confusion_matrix(y,pred,labels=[0,1])
    return dict(n=int(len(y)),positive_count=int(sum(y)),negative_count=int(len(y)-sum(y)),
                precision=float(p),recall=float(r),f1=float(f),accuracy=float(accuracy_score(y,pred)),
                confusion_matrix=cm.tolist(),matrix_order='rows=true [0,1]; columns=predicted [0,1]')

def evaluate(model_path=None,output_dir=None,overwrite=False):
    model_path=Path(model_path) if model_path else ROOT/'artifacts/model.joblib'
    out=Path(output_dir) if output_dir else ROOT/'results'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() and not overwrite:
        raise FileExistsError('Evaluation already saved. Use a separate --output-dir for reproduction.')
    df=pd.read_csv(ROOT/'data/labeled.csv',keep_default_na=False)
    test=df[df.split.eq('test')].copy()
    bundle=joblib.load(model_path);p=bundle['pipeline'].predict_proba(test.text.map(normalize))[:,1]
    test['probability']=p;test['prediction']=(p>=bundle['threshold']).astype(int)
    test['decision_score']=bundle['pipeline'].decision_function(test.text.map(normalize))
    test['correct']=test.label.eq(test.prediction)
    result=metrics(test.label.to_numpy(),test.prediction.to_numpy())
    result.update(threshold=float(bundle['threshold']),dataset_sha256=sha256(ROOT/'data/labeled.csv'),
                  model_sha256=sha256(model_path),evaluation_scope='Source-selected real reviews plus AI-authored contrast pairs; not a representative marketplace benchmark',
                  subgroups={})
    for name,mask in [('real_only',test.is_synthetic.eq(0)),('synthetic_only',test.is_synthetic.eq(1)),
                      ('new_test_only',test.evaluation_cohort.eq('new_test')),
                      ('new_real_only',test.evaluation_cohort.eq('new_test')&test.is_synthetic.eq(0)),
                      ('legacy_test',test.evaluation_cohort.eq('legacy_test')),
                      ('real_marketplace',test.is_synthetic.eq(0)&test.domain.eq('market')),
                      ('real_appstore',test.is_synthetic.eq(0)&test.domain.eq('appstore'))]:
        subset=test[mask]
        if len(subset):
            m=metrics(subset.label.to_numpy(),subset.prediction.to_numpy())
            if subset.label.sum()==0:
                m.update(precision=None,recall=None,f1=None,note='No positive labels: class-1 metrics are undefined; inspect false-positive count instead.')
            result['subgroups'][name]=m
    result['primary_evaluation']='new_test_only: excludes every example available in v1, whose test errors were previously inspected'
    test.to_csv(out/'test_predictions.csv',index=False)
    test[~test.correct].to_csv(out/'errors.csv',index=False)
    write_json(out/'metrics.json',result)
    fresh=test[test.evaluation_cohort.eq('new_test')]
    fresh.to_csv(out/'new_test_predictions.csv',index=False)
    write_json(out/'new_test_metrics.json',result['subgroups']['new_test_only'])
    cm=np.array(result['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(5.5,4.5),layout='constrained')
    ax.imshow(cm,cmap='Blues')
    for (i,j),value in np.ndenumerate(cm):
        ax.text(j,i,str(value),ha='center',va='center',fontsize=18,color='white' if value>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['0: clean','1: profanity'],yticklabels=['0: clean','1: profanity'],xlabel='Predicted label',ylabel='True label',title=f'Held-out test: n={len(test)}, F1={result["f1"]:.3f}')
    fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    fresh_metrics=result['subgroups']['new_test_only']
    fresh_cm=np.array(fresh_metrics['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(5.5,4.5),layout='constrained')
    ax.imshow(fresh_cm,cmap='Blues')
    for (i,j),value in np.ndenumerate(fresh_cm):
        ax.text(j,i,str(value),ha='center',va='center',fontsize=18,color='white' if value>fresh_cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['0: clean','1: profanity'],yticklabels=['0: clean','1: profanity'],xlabel='Predicted label',ylabel='True label',title=f'New held-out test: n={len(fresh)}, F1={fresh_metrics["f1"]:.3f}')
    fig.savefig(out/'confusion_matrix_new_test.png',dpi=180);plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--overwrite',action='store_true',help='Only for rerunning the same frozen model, never for test-guided tuning.')
    args=parser.parse_args();evaluate(args.model,args.output_dir,args.overwrite)
