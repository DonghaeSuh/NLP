# 컨테이너에서 실행  네이버 말뭉치 다운
# git pull origin master
# bash preprocess.sh dump-raw-nsmc    

#전처리
# bash preprocess.sh process-nsmc


import math
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer

corpus_fname ="/notebooks/embedding/data/processed/processed_ratings.txt"
model_fname="/notebooks/embedding/data/processed/soyword.model"

sentences =[sent.strip() for sent in open(corpus_fname,'r').readlines()]
word_extractor =WordExtractor(min_frequency=100,min_cohesion_forward=0.05,min_right_branching_entropy=0.0)
word_extractor.train(sentences)
word_extractor.save(model_fname)

word_extractor.load(model_fname)
scores =word_extractor.word_scores()
scores={key:(scores[key].cohesion_forward*math.exp(scores[key].right_branching_entropy)) for key in scores.keys()}
tokenizer =LTokenizer(scores=scores)
tokens =tokenizer.tokenize("에비는 종이었다")
tokens