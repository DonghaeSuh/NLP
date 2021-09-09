"""
데이터 다운로드
git pull origin master 
bash preprocess.sh dump-blog
"""

from prerpocess import get_tokenizer 

corpus_fname = "data/processed/processed_blog.txt"

tokenizer = get_tokenizer("mecab")
titles,raw_corpus, noun_corpus = [],[],[]
with open(corpus_fname,'r',encoding='utf-8') as f:
    for line in f:
        try:
            title,document = line.strip().split("\u241E")
            titles.append(title)
            raw_corpus.append(document)
            nouns = tokenizer.nouns(document)
            noun_corpus.append(' '.join(nouns))
        except:
            continue

#TF-IDF

from sklearn.feature_extraction.text import TfidVectorizer
vectorizer = TfidVectorizer(
    min_df = 1,
    ngram_range=(1,1),
    lowercase = True,
    tokenizer = lambda x: x.split())
input_matrix = vectorizer.fit_transform(noun_corpus)


#TF-IDF 결과 확인

id2vocab = {vectorizer.vocabulary_[token]:token for token in vectorizer.vocabulary_.keys()}
#curr_doc : 말뭉치 첫 번쨰 문서의 TF-IDF 벡터
curr_doc, result = input_matrix[0], []
#curr_doc에서 TF-IDF 값이 0이 아닌 요소들을 내림차순 정렬
for idx, el in zip(curr_doc.indices, curr_doc.data):
    result.append((id2vocab[idx], el))
sorted(result, key=lambda x: x[1], reverse=True)

#100차원 SVD 차원축소

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components =100)
vecs = svd.fit_transform(input_matrix)


#유사 문서 검색
from models.sent_eval import LSAEvaluator
model = LSAEvaluator("data/entence-embeddings/lsa-tfidf/lsa-tfidf.vecs")
model.most_similar(doc_id=0)  #md 문서 제목명에 해당하는 문서 임베딩과 코사인 유사도가 높은 문서

#LSA 시각화 '품질나쁨주의' 
model.visualize("between")
model.visualize("tsne")