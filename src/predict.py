"""Load the saved model without retraining and predict one custom comment."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from .common import ROOT,normalize

def predict(text,model_path=None,explain=False):
    cleaned=normalize(text)
    if not cleaned:raise ValueError('Enter a nonempty comment.')
    bundle=joblib.load(model_path or ROOT/'artifacts/model.joblib')
    pipeline=bundle['pipeline'];threshold=float(bundle['threshold'])
    probability=float(pipeline.predict_proba([cleaned])[0,1])
    score=float(pipeline.decision_function([cleaned])[0])
    result=dict(prediction=int(probability>=threshold),profanity_probability=probability,
                decision_score=score,threshold=threshold)
    if explain:
        vectorizer=pipeline.named_steps['tfidf'];classifier=pipeline.named_steps['classifier']
        x=vectorizer.transform([cleaned]).tocsr()
        terms=vectorizer.get_feature_names_out()[x.indices]
        contributions=x.data*classifier.coef_[0,x.indices]
        indices=np.argsort(np.abs(contributions))[::-1][:10]
        result['normalized_text']=cleaned
        result['top_ngram_contributions']=[dict(ngram=str(terms[i]),logit_contribution=float(contributions[i])) for i in indices]
        result['intercept']=float(classifier.intercept_[0])
        result['known_ngram_count']=int(x.nnz)
        result['note']='Overlapping n-gram contributions explain this linear score; they are not causal explanations.'
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text',required=True)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--explain',action='store_true')
    args=parser.parse_args()
    print(json.dumps(predict(args.text,args.model,args.explain),ensure_ascii=False,indent=2))
