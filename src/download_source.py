"""Optional provenance check: download the pinned source and verify selected records."""
import hashlib
import io
import json
import urllib.request
import zipfile
import pandas as pd
from .common import ROOT

def main():
    manifest=json.loads((ROOT/'data/source_manifest.json').read_text())
    url=f'https://raw.githubusercontent.com/IS2AI/KazSAnDRA/{manifest["source_commit"]}/{manifest["archive"]}'
    request=urllib.request.Request(url,headers={'User-Agent':'Kazakh-Profanity-Research/1.0'})
    with urllib.request.urlopen(request,timeout=120) as response:content=response.read()
    if hashlib.sha256(content).hexdigest()!=manifest['archive_sha256']:
        raise ValueError('Source archive checksum mismatch; do not silently use a changed dataset.')
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        original=pd.read_csv(z.open(manifest['csv_member']))
    annotations=pd.read_csv(ROOT/'data/base_annotations.csv',keep_default_na=False)
    real=annotations[annotations.is_synthetic.eq(0)]
    by_id=original.set_index('custom_id')
    for row in real.itertuples():
        assert row.source_id in by_id.index
        assert row.text==by_id.loc[row.source_id,'text']
    print(f'Archive checksum and all {len(real)} real source texts verified.')
    print('Original sentiment labels are not used as profanity labels.')

if __name__=='__main__':main()
