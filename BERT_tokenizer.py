import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Bert-base의 토크나이저

result = tokenizer.tokenize('Here is the sentence I want embeddings for.')
print(result)

# ['here', 'is', 'the', 'sentence', 'i', 'want', 'em', '##bed', '##ding', '##s', 'for', '.']

print(tokenizer.vocab['here'])
# 2182 : here 가 2182라는 정수로 맵핑되어있음

print(tokenizer.vocab['embeddings'])
# KeyError: 'embeddings' : 존재하지 않는 단어 OOV 

print(tokenizer.vocab['em'])     # 7861
print(tokenizer.vocab['##bed'])  # 8270
print(tokenizer.vocab['##ding']) # 4667
print(tokenizer.vocab['##s'])    # 2015

# BERT의 단어 집합을 vocabulary.txt에 저장
with open('vocabulary.txt', 'w') as f:
  for token in tokenizer.vocab.keys():
    f.write(token + '\n')

# 쉼표로 구분 된 값 (csv) 파일을 DataFrame으로 읽음
df = pd.read_fwf('vocabulary.txt', header=None)
df

print('단어 집합의 크기 :',len(df))
# 단어 집합의 크기 : 30522

df.loc[4667].values[0]
# ##ding

# 특수 토큰
# [PAD] = 0
#[UNK] = 100
#[CLS] = 101
#[SEP] = 102
#[MASK] = 103