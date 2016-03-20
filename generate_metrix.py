#!/usr/bin/python
import sys
from gensim.models import word2vec
import tensorflow as tf
import numpy as np
import cPickle
import argparse

def load_w2v_model(w2v_file):
    return word2vec.Word2Vec.load_word2vec_format(w2v_file, binary=True)

def generate_feature(sentence_file, w2v, w2vsize):
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
                    #set default value if the word is not in w2v dictionary
                    feature.append([0.00001] * w2vsize)
            sen.append(terms)
            fmatrix.append(feature)
    return sen, fmatrix

if __name__=="__main__": 

    parser = argparse.ArgumentParser(description='Generate feature vector for sentences')
    parser.add_argument('-w2v', help='word2vector Binary File', required=True)
    parser.add_argument('-w2v_size', help='word2vector size (default: 300)', default=300)
    parser.add_argument('-s', '--sentences', help='input sentence file', required=True)
    parser.add_argument('-l', '--label', type=int, choices=[0, 1], help='label of the data, [0: negative, 1: positive]', required=True)
    parser.add_argument('-v', '--sentence_vectors', help='output sentence vectors', required=True)

    args = parser.parse_args()

    w2v = load_w2v_model(args.w2v)

    sen, fmatrix = generate_feature(args.sentences, w2v, args.w2v_size)
    cPickle.dump([sen, fmatrix, [args.label]*len(sen)], open(args.sentence_vectors, 'w'))

