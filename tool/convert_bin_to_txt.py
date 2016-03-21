#!/usr/bin/python
import sys
from gensim.models import word2vec

fname = sys.argv[1]

model = word2vec.Word2Vec.load_word2vec_format(fname, binary=True)
print model['apple']
#model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)

