#!/usr/bin/python
import sys
from gensim.models import word2vec
import tensorflow as tf
import numpy as np
import cPickle


if __name__=="__main__": 

    sen, fmatrix, label = cPickle.load(open(sys.argv[1], 'r'))
    print len(sen)
    print len(fmatrix)
    print len(label)
    print fmatrix
