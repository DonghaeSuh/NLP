#전처리 데이터 다운로드
"""
git pull origin master
bash preprocess.sh dump-processed
"""

#LDA model feature 생성, 단어 등장순서 고려하지 않고 빈도만을 따짐.
from gensim import corpora
from preprocess import get_tokenizer

corpus_fname ="/notebooks/embedding/data/processed/corrected_ratings_corpus.txt"

documets, tokenizer_corpus = [], []
tokenizer =get_tokenizer("mecab")

with open(corpus_fname, 'r', encoding ='utf-8') as f:
    for document in f:
        tokens =list(set(tokenizer.morphs(document.strip())))
        documents.append(documet)
        tokenized_corpus.append(tokens)
dictionary = corpora.Dictionary(tokenized_corpus)
corpus = [dictionary.doc2bow(text) for text in tokenized_corpus]


#LDA 학습 및 결과 확인
from gensim.models import ldamulticore
LDA = ldamulticore.LdaMulticore(corpus, id2word=dictionary, dum_topics=30, workers=4)
all_topics =LDA.get_document_topics(corpus, minimum_probability=0.5, per_word_topics=False)
for doc_idx, topic in enumerate(all_topics[:5]):
    print(doc_idx, topic)


"""
LDA 학습 스크립트 (bash 셀 내에서 작성)
python models/sent_utils.py --method latent_dirichlet_allocation \
    --input_path /notebooks/embedding/data/processed/corrected_ratings_corpus.txt \
    -- output_path /notebooks/embedding/data/sentence-embeddings/lda/lda
"""

#LDA 평가 모듈 선언
from models.sent_eval import LDAEvaluator
model =LDAEvaluator("data/sentence-embeddings/lda/lda")

#토픽별 문서 확인  => 해당 토픽 ID에서 가장 높은 확률 값을 지니는 단어들 목록을 확인
model.show_topic_docs(topic_id=0)

#토픽별 단어 확인 =>토픽의 단어 분포를 확인
model.show_topic_words(topic_id=0)

#새로운 문서의 토픽 확인 => 토픽 id와 확률값 리턴
model.show_new_document_topic(["너무 사랑스러운 영화", "인생을 말하는 영화"])

