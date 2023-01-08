#!/usr/bin/env python

# Supress warnings
import os
import time
import warnings

import torch
from sqlalchemy import MetaData, Table, create_engine, inspect, update
from sqlalchemy.sql import select
from transformers import MarianMTModel, MarianTokenizer

warnings.filterwarnings("ignore")

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Which pairid value to start translating
startpairid = 0

def translate(text, model, tokenizer):
    print(" Removing newlines ...  ")
    print(" Number of words: ", len(text.split()))
    text = text.replace('\n\n', '\n')
    sentences = text.split('\n')
    print(" Number of paragraphs: ", len(sentences))

    print("Sentences: ", sentences)
    translation_output=[]
    for i in sentences:
        print(" Sentence: ", i)
        sentencelength=len(i.split())
        if sentencelength > 300:
            print(" Sentence too long, splitting into chunks ...  ")
            chunks = [i[x:x+500] for x in range(0, len(i), 500)]
            print(" Number of chunks: ", len(chunks))
            print(" Chunks: ", chunks)
            for c in chunks:
                print(" Chunk: ", c)
                translated = model.generate(
                    **tokenizer(c, return_tensors="pt", padding=True).to(device))
                print("Decoding...")
                translation_output.append([tokenizer.decode(t, skip_special_tokens=True) for t in translated][0])
        else:
            print(" Sentence short enough to translate as is ...  ")
            translated = model.generate(
                **tokenizer(i, return_tensors="pt", padding=True).to(device))
            print("Decoding...")
            translation_output.append([tokenizer.decode(t, skip_special_tokens=True) for t in translated][0])
    print(translation_output)
    return "\n".join(translation_output)

if torch.cuda.is_available():
    dev = "cuda"
    print("Using CUDA")
elif torch.has_mps:
    dev = "cpu"
    # torch.device = "cpu"
    print("Using Apple MPS")
else:
    dev = "cpu"
    print("Using CPU")

device = torch.device(dev)

print("Loading models ...  ")
fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_en_model.to(device)

en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_fr_model.to(device)

en_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_es_model.to(device)

fr_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-es")
fr_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-es")
fr_es_model.to(device)

dbcon = create_engine(
    'sqlite:////Users/scottsyms/code/HeritageCanada/data/sitecontent2.db')

metadata = MetaData()
source = Table(
    'source', metadata, autoload=True, autoload_with=dbcon)

print("Beginning translation ...  ")
# Modify English
result = dbcon.execute(select(
    [source.c.id, source.c.language, source.c.text]).where(source.c.pairid > startpairid))
    
     #.limit(5))

selectid = []
[selectid.append(i) for i in result]

count = 0
for i in selectid:
    print("Translating row: " + str(i[0]))
    # If language is en
    if i[1] == 'en':
        movelanguage = source.update().values(english=i[2]).where(
            source.c.id == selectid[count][0])
        print(" Translate to French")
        translatealternate = source.update().values(french=translate(i[2], en_fr_model, en_fr_tokenizer)).where(
            source.c.id == selectid[count][0])
        print(" Translate to Spanish")
        translatetospanish = source.update().values(spanish=translate(i[2], en_es_model, en_es_tokenizer)).where(
            source.c.id == selectid[count][0])
    elif i[1] == 'fr':
        movelanguage = source.update().values(french=i[2]).where(
            source.c.id == selectid[count][0])
        translatealternate = source.update().values(english=translate(i[2], fr_en_model, fr_en_tokenizer)).where(
            source.c.id == selectid[count][0])
        translatetospanish = source.update().values(spanish=translate(i[2], fr_es_model, fr_es_tokenizer)).where(
            source.c.id == selectid[count][0])
    dbcon.execute(movelanguage)
    dbcon.execute(translatealternate)
    dbcon.execute(translatetospanish)
    count += 1


print("Done")
