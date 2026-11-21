"""Fit on train only; select C, class weighting and threshold on validation only."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from .common import ROOT, load_config, normalize, sha256, write_json

def train(output_dir=None):
    cfg=load_config()
    out=Path(output_dir) if output_dir else ROOT/'artifacts'
    out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/labeled.csv',keep_default_na=False)
    tr=data[data.split.eq('train')]; va=data[data.split.eq('validation')]
    assert set(tr.group_id).isdisjoint(va.group_id)
    x_train=tr.text.map(normalize); x_val=va.text.map(normalize)
    candidates=[];best=None;best_key=None
    for c in cfg['C_grid']:
        for weight in cfg['class_weight_grid']:
            pipeline=Pipeline([
                ('tfidf',TfidfVectorizer(analyzer='char',ngram_range=tuple(cfg['ngram_range']),min_df=cfg['min_df'],
                    max_features=cfg['max_features'],sublinear_tf=True,lowercase=False)),
                ('classifier',LogisticRegression(C=c,class_weight=weight,solver='liblinear',max_iter=2000,random_state=cfg['seed']))])
            pipeline.fit(x_train,tr.label)
            scores=pipeline.predict_proba(x_val)[:,1]
            for threshold in cfg['threshold_grid']:
                pred=(scores>=threshold).astype(int)
                p,r,f,_=precision_recall_fscore_support(va.label,pred,average='binary',zero_division=0)
                record=dict(C=c,class_weight=weight,threshold=threshold,precision=float(p),recall=float(r),f1=float(f))
                candidates.append(record)
                # Deterministic tie-break: recall, precision, threshold near .5, smaller C.
                key=(f,r,p,-abs(threshold-.5),-c,weight is None)
                if best_key is None or key>best_key:
                    best_key=key;best=(pipeline,record)
    pipeline,selected=best
    bundle=dict(pipeline=pipeline,threshold=selected['threshold'],positive_label=1,
                normalization='src.common.normalize',policy_version='1.0',seed=cfg['seed'])
    joblib.dump(bundle,out/'model.joblib',compress=3)
    pd.DataFrame(candidates).to_csv(out/'validation_search.csv',index=False)
    vocab=pipeline.named_steps['tfidf'].get_feature_names_out()
    coefs=pipeline.named_steps['classifier'].coef_[0]
    order=np.argsort(coefs)
    pd.DataFrame({'ngram':vocab[np.r_[order[:30],order[-30:]]],'coefficient':coefs[np.r_[order[:30],order[-30:]]]}).to_csv(out/'feature_coefficients.csv',index=False)
    metadata=dict(model=cfg['model'],selected=selected,train_examples=len(tr),validation_examples=len(va),
                  vocabulary_size=len(vocab),seed=cfg['seed'],sklearn_version=sklearn.__version__,
                  dataset_sha256=sha256(ROOT/'data/labeled.csv'),config_sha256=sha256(ROOT/'config.json'),
                  model_sha256=sha256(out/'model.joblib'),
                  training_objective='L2-regularized binary logistic loss; C is inverse regularization strength',
                  validation_selection='Max class-1 F1; ties: recall, precision, threshold close to 0.5, smaller C, no weights',
                  test_used_for_selection=False,calibrated_for_marketplace_prevalence=False)
    write_json(out/'training_metadata.json',metadata)
    print('Selected on VALIDATION only:',selected)
    print('Vocabulary size:',len(vocab),'Saved:',out/'model.joblib')
    return bundle

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();train(args.output_dir)
