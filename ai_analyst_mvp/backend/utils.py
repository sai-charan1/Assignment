import os
from pathlib import Path
from typing import List
from pypdf import PdfReader
import re

def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(path:Path)->str:
    reader = PdfReader(str(path))
    texts=[]
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception as e:
            texts.append("")
    return "\n".join(texts)

def clean_text(s:str)->str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def chunk_text_semantic(text:str, min_size=200, max_size=800, overlap=50):
    # simplistic semantic chunker by paragraphs and sentence counts
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(text)
    chunks=[]
    current=[]
    curr_len=0
    for sent in sents:
        sent_len = len(sent.split())
        if curr_len + sent_len > max_size and curr_len >= min_size:
            chunks.append(' '.join(current))
            # start overlap
            if overlap>0:
                current = current[-overlap:]
                curr_len = sum(len(s.split()) for s in current)
            else:
                current=[]
                curr_len=0
        current.append(sent)
        curr_len += sent_len
    if current:
        chunks.append(' '.join(current))
    return [clean_text(c) for c in chunks if len(c.strip())>20]
