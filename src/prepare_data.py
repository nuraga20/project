"""Deduplicate, group near-duplicates, and assign reproducible grouped splits."""
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics.pairwise import cosine_similarity
from .common import ROOT, normalize, duplicate_key, load_config, sha256, write_json

def prepare():
    cfg=load_config()
    df=pd.read_csv(ROOT/'data/base_annotations.csv',keep_default_na=False)
    assert df.id.is_unique and set(df.label)=={0,1}
    df['normalized_text']=df.text.map(normalize)
    assert df.normalized_text.str.len().gt(0).all()
    # Exact normalized text matches are dropped only if their annotations agree.
    conflict=df.groupby('normalized_text').label.nunique()
    if conflict.gt(1).any(): raise ValueError('Conflicting duplicate annotations')
    removed=df[df.normalized_text.duplicated()].copy()
    removed.to_csv(ROOT/'data/removed_duplicates.csv',index=False)
    df=df.drop_duplicates('normalized_text').sort_values('id').reset_index(drop=True)
    parent=list(range(len(df)))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for col in ['group_id']:
        for indices in df.groupby(col).indices.values():
            for j in indices[1:]:union(int(indices[0]),int(j))
    keys={}
    for i,t in enumerate(df.text):
        k=duplicate_key(t)
        if k in keys:union(i,keys[k])
        else:keys[k]=i
    # Fixed hashing has no fitted vocabulary/IDF. It is used ONLY for deduplication.
    hv=HashingVectorizer(analyzer='char',ngram_range=(3,5),n_features=2**18,alternate_sign=False,norm='l2',lowercase=False)
    sim=cosine_similarity(hv.transform(df.normalized_text))
    pairs=np.argwhere(np.triu(sim>=cfg['near_duplicate_threshold'],k=1))
    audit=[]
    for i,j in pairs:
        union(int(i),int(j))
        audit.append(dict(id_a=df.iloc[i].id,id_b=df.iloc[j].id,similarity=float(sim[i,j])))
    df['group_id']=[f'group_{find(i):04d}' for i in range(len(df))]
    cv=StratifiedGroupKFold(n_splits=cfg['split_folds'],shuffle=True,random_state=cfg['seed'])
    fold=np.full(len(df),-1)
    for k,(_,idx) in enumerate(cv.split(df,df.label,df.group_id)):fold[idx]=k
    df['split']=np.where(fold==cfg['test_fold'],'test',np.where(fold==cfg['validation_fold'],'validation','train'))
    # Preserve prior holdouts when expanding. Priority may move development examples
    # into test if a new near-duplicate connects them, never the reverse.
    prior_path=ROOT/'data/previous_split_membership.csv'
    prior=pd.read_csv(prior_path) if prior_path.exists() else pd.DataFrame(columns=['id','split'])
    prior_map=dict(zip(prior.id,prior.split))
    priority={'train':0,'validation':1,'test':2}
    for group,idx in df.groupby('group_id').indices.items():
        old_splits=[prior_map[df.iloc[i].id] for i in idx if df.iloc[i].id in prior_map]
        if old_splits:df.loc[idx,'split']=max(old_splits,key=lambda s:priority[s])
    df['evaluation_cohort']=[('legacy_test' if prior_map.get(r.id)=='test' else
                             'previously_development' if r.id in prior_map else 'new_test')
                            if r.split=='test' else 'development' for r in df.itertuples()]
    assert set(prior[prior.split.eq('test')].id).issubset(set(df[df.split.eq('test')].id))
    assert df.groupby('group_id').split.nunique().max()==1
    assert all(set(df[df.split.eq(s)].label)=={0,1} for s in ['train','validation','test'])
    for i,j in pairs:assert df.iloc[i].split==df.iloc[j].split
    df.to_csv(ROOT/'data/labeled.csv',index=False)
    df[['id','group_id','split']].to_csv(ROOT/'data/splits.csv',index=False)
    pd.DataFrame(audit,columns=['id_a','id_b','similarity']).to_csv(ROOT/'data/near_duplicate_pairs.csv',index=False)
    counts=df.groupby(['split','is_synthetic','label']).size().reset_index(name='count').to_dict('records')
    split_values=df.split.to_numpy()
    cross_mask=split_values[:,None]!=split_values[None,:]
    summary=dict(seed=cfg['seed'],total=len(df),exact_duplicates_removed=len(removed),groups=df.group_id.nunique(),
                 near_duplicate_pairs=len(audit),near_duplicate_threshold=cfg['near_duplicate_threshold'],
                 max_cross_split_similarity=float(sim[cross_mask].max()),
                 split_counts=df.split.value_counts().to_dict(),class_counts={str(k):int(v) for k,v in df.label.value_counts().items()},
                 breakdown=counts,real_count=int(df.is_synthetic.eq(0).sum()),synthetic_count=int(df.is_synthetic.eq(1).sum()),
                 test_cohorts=df[df.split.eq('test')].evaluation_cohort.value_counts().to_dict(),
                 previous_test_preserved=True,base_annotations_sha256=sha256(ROOT/'data/base_annotations.csv'))
    write_json(ROOT/'data/dataset_summary.json',summary)
    print(df.groupby(['split','is_synthetic','label']).size().to_string())
    print('Groups:',summary['groups'],'maximum cross-split similarity:',summary['max_cross_split_similarity'])
    return df

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
