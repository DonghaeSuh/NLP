# 전처리 데이터 다운로드
"""
git pull origin master
bash preprocess.sh dump-processed
"""
#학습 데이터 구축
#bash sentimodel.sh preprocess-nsmc

#Doc2Vec 입력 클래스
from preprocess import get_tokenizer
from gensim.models.doc2vec import TaggedDocument

class Doc2VecInput:

    def __init__(self,fname,tokenizer_name = "mecab"):
        self.fname =fname
        self.tokenizer = get_tokenizer(tokenizer_name)

    def __init__(self):
        with open(self.fname, encoding='utf-8') as f:
            for line in f:
                try:
                    sentence,movie_id =line.strip().split("\u241E")
                    tokens = self.tokenizer.morphs(sentence)
                    tagged_doc = TaggedDocument(words=tokens, tag=['MOVIE_%s' % movie_id])
                    yield tagged_doc
                except:
                    continue


#Doc2Vec 학습
from gensim.models import Doc2Vec

corpus_fname = "/notebooks/embedding/data/processed/processed_review_movieid.txt"
output_fname = "/notebooks/embedding/data/sentence-embeddings/doc2vec/doc2vec.model"
corpus = Doc2VecInput(corpus_fname)
model =Doc2Vec(corpus, dm=1, vector_size=100)
model.save(output_fname)


