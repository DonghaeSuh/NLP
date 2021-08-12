data =tokenizer.analyze("아버지가방에들어가신다")
tokens=[]
for word in data:
    tokens.extend([str(m).split("/")[0] for m in word.morphs])