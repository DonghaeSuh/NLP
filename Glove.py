# Glcve 컨테이너 작성 

cd /notebooks/embedding
mkdir -p data/word-embeddings/glove

models/glove/build/vocab_count -min-count 5 -verbose 2 < data/tokenized/corpus_mecab.txt > data/word-embeddings/glove/glove.vocab
models/glove/build/cooccur -memory 10.0 -vocab-file data/word-embeddings/glove/glove.vocab -verbose 2 -window-size 15 < data/tokenized/corpus_mecab.txt > data/word-embeddings/glove/glove.cooc

models/glove/build/shuffle -memory 10.0 -verbose 2 < data/word-embeddings/glove/glove.cooc > data/word-embeddings/glove/glove.shuf

models/glove/build/glove -save-file data/word-embeddings/glove/glove
-threads 4 -input-file data/word-embeddings/glove/glove.shuf -x-masx 10 -iter 15 -vector-size 100 -binary 2 -vocab-file data/word-embeddings/glove/glove.vocab -verbose 2