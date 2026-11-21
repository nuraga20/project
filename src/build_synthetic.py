"""Reproduce the 408 explicitly templated additions. Original 192 authored rows stay fixed."""
import json
import pandas as pd
from .common import ROOT

def build():
    spec=json.loads((ROOT/'data/synthetic_scenarios.json').read_text())
    rows=[]
    for i,scenario in enumerate(spec['scenarios']):
        for j,product in enumerate(spec['products']):
            for label,key in [(0,'clean'),(1,'profane')]:
                rows.append(dict(id=f'synthetic_v2_{i:02d}_{j}_{label}',text=scenario[key].format(product=product),
                    label=label,source='AI-authored templated product-review contrast pairs',
                    source_id=f'v2_template_{i:02d}_product_{j}',domain='product_review_constructed',
                    is_synthetic=1,group_id=f'v2_template_{i:02d}',
                    annotation_method='AI-authored templates and deterministic substitution; no independent human verification',
                    label_note='Constructed example: '+('Kazakh profanity present' if label else 'no identifiable Kazakh profanity'),
                    selection='template_substitution'))
    df=pd.DataFrame(rows)
    assert len(df)==408 and df.text.nunique()==408
    return df

if __name__=='__main__':
    df=build()
    df.to_csv(ROOT/'data/synthetic_additions.csv',index=False)
    print('Created 408 marked synthetic additions; 34 template-family groups.')
