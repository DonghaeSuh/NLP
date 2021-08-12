from konlpy.tag import Okt,Komoran,Mecab,Hannanum,Kkma

def get_tokenizer(tokenizer_name):
    if tokenizer_name =="komoran":
        tokenizer=Komoran()
    elif tokenizer_name =="okt":
        tokenizer =Okt()
    elif tokenizer_name =="mecab":
        tokenizer =Mecab()
    elif tokenizer_name =="hannanum":
        tokenizer =hannanum()
    elif tokenizer_name =="kkma":
        tokenizer =Kkma()
    else:
        tokenizer =Mecab()
    return tokenizer

# 코모란 사용 예시
tokenizer =get_tokenizer("komoran")
tokenizer.morphs("아버지가방에들어가신다")
tokenizer.pos("아버지가방에들어가신다")
