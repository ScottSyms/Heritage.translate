#!/usr/bin/env python

# Supress warnings
from sqlalchemy import create_engine, MetaData, Table, update, inspect
from sqlalchemy.sql import select
from transformers import MarianMTModel, MarianTokenizer
import torch
import time
import warnings
warnings.filterwarnings("ignore")


def translate(text, model, tokenizer):
    print(" Removing newlines ...  ")
    print(" Number of words: ", len(text.split()))
    text = text.replace('\n\n', '\n')
    sentences = text.split('\n')
    print(" Number of paragraphs: ", len(sentences))


    # print("Sentences: ", sentences)
    translated = model.generate(
        **tokenizer(sentences, return_tensors="pt", padding=True))
    print("Decoding...")
    test = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return "\n".join(test)


if torch.cuda.is_available():
    torch.device = "cuda"
    print("Using CUDA")
elif torch.has_mps:
    torch.device = "mps"
    torch.device = "cpu"
    print("Using Apple MPS")
else:
    torch.device = "cpu"
    print("Using CPU")

print("Loading models ...  ")
fr_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_en_model.to(torch.device)

en_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_fr_model.to(torch.device)

en_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
en_es_model.to(torch.device)

fr_es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-es")
fr_es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-es")
fr_es_model.to(torch.device)


dbcon = create_engine(
    'sqlite:////Users/scottsyms/code/HeritageCanada/data/fish/sample2.db')

metadata = MetaData()
source = Table(
    'source', metadata, autoload=True, autoload_with=dbcon)

print("Beginning translation ...  ")
# Modify English
result = dbcon.execute(select(
    [source.c.id, source.c.language, source.c.text]).limit(5))

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
        translatetospanish = source.update().values(spanish='Translation to Spanish').where(
            source.c.id == selectid[count][0])
    elif i[1] == 'fr':
        movelanguage = source.update().values(french=i[2]).where(
            source.c.id == selectid[count][0])
        translatealternate = source.update().values(english='Translation to English').where(
            source.c.id == selectid[count][0])
        translatetospanish = source.update().values(spanish='Translation to Spanish').where(
            source.c.id == selectid[count][0])
    dbcon.execute(movelanguage)
    dbcon.execute(translatealternate)
    dbcon.execute(translatetospanish)
    count += 1


print("Done")
