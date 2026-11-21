"""Shared text handling. These functions must match training and inference."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def normalize(text):
    if not isinstance(text, str):
        raise TypeError('text must be a string')
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r'[\w.+-]+@[\w.-]+\.[a-z]{2,}', ' [email] ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' [url] ', text)
    text = re.sub(r'(?<!\w)@[\w.]+', ' [user] ', text)
    text = re.sub(r'(?<!\w)\+?[\d][\d ()-]{8,}[\d](?!\w)', ' [number] ', text)
    return re.sub(r'\s+', ' ', text).strip()

def duplicate_key(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', normalize(text))).strip()

def load_config():
    return json.loads((ROOT/'config.json').read_text())

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
