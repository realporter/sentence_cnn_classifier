#!/usr/bin/python
import sys
from gensim.models import word2vec
import tensorflow as tf
import numpy as np
import cPickle

def load_w2v_model(w2v_file):
    return word2vec.Word2Vec.load_word2vec_format(w2v_file, binary=True)

def generate_feature(sentence_file, w2v, w2vsize=300):
    sen = []
    fmatrix = []
    with open(sentence_file, 'r') as INFILE:
        for line in INFILE:
            terms = line.strip().split(' ')
            feature = []
            for term in terms:
                if term in w2v:
                    feature.append(w2v[term].tolist())
                else:
                    feature.append([0.0] * w2vsize)
            sen.append(terms)
            fmatrix.append(feature)
    return sen, fmatrix

if __name__=="__main__": 

    w2v_file = sys.argv[1]
    sentence_file = sys.argv[2]
    db_file = sys.argv[3]
    label = int(sys.argv[4])

    w2v = load_w2v_model(w2v_file)

    sen, fmatrix = generate_feature(sentence_file, w2v)
    cPickle.dump([sen, fmatrix, [label]*len(sen)], open(db_file, 'w'))

