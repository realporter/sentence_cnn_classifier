#!/usr/bin/python
import sys
import tensorflow as tf
import numpy as np
import cPickle
import random
import util.cnn_utils as cu
import argparse


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate feature vector for sentences')
    parser.add_argument('-pf', '--pos_file', help='positive feature vectors for training', required=True)
    parser.add_argument('-nf', '--neg_file', help='negitive feature vectors for training', required=True)
    parser.add_argument('-w2v_size', type=int, help='word2vector size (default: 300)', default=300)
    parser.add_argument('--label_size', type=int, help='how many classes? (default: 2)', default=2)
    parser.add_argument('-m', '--model_input', help='the trained model to read', required=True)
    parser.add_argument('-slt', '--max_sen_len_train', type=int, help='the max sentence length of training data', required=True)
    args = parser.parse_args()

    model_name = args.model_input
    w2v_size = args.w2v_size
    label_size = args.label_size

    ###load sentence with w2v word embeddings
    pos_sen, pos_fmatrix, pos_label = cPickle.load(open(args.pos_file, 'r'))
    neg_sen, neg_fmatrix, neg_label = cPickle.load(open(args.neg_file, 'r'))

    ###build and shuffle data
    sentences_test, labels_test, raw_sen_test = cu.build_data(pos_fmatrix, neg_fmatrix, pos_label, neg_label, pos_sen, neg_sen)

    ###pad sentences to the same length
    max_length = max(len(s) for s in (pos_fmatrix + neg_fmatrix))
    print 'test max_length: %d' % max_length
    print 'train max_length: %d' % args.max_sen_len_train
    if max_length < args.max_sen_len_train:
        max_length = args.max_sen_len_train
    cu.pad_sentence(sentences_test, w2v_size, max_length)

    ##########################
    ### Construct the grah ###
    ##########################

    ###training data placeholder
    with tf.name_scope('feature-vectors') as scope:
        x = tf.placeholder("float", shape=[None, max_length, w2v_size])
    with tf.name_scope('labels') as scope:
        y_ = tf.placeholder("float", shape=[None, label_size])

    ###Convolution and Pooling
    feature_size1 = 100
    filter_list = [3, 4, 5]

    poolings = []

    for idx, filter_size in enumerate(filter_list):
        with tf.name_scope('conv-window-size-%d' % filter_size) as scope:
            x_image = tf.reshape(x, shape=[-1, max_length, w2v_size, 1])
            W_conv = cu.weight_variable([filter_size, w2v_size, 1, feature_size1])
            b_conv = cu.bias_variable([feature_size1])
            h_conv = tf.nn.relu(cu.conv2d(x_image, W_conv) + b_conv)
        with tf.name_scope('pool-window-size-%d' % filter_size) as scope:
            h_pool = cu.max_pool_Nx1(h_conv, max_length - filter_size + 1)
        poolings.append(h_pool)

    ###combine pooled features
    with tf.name_scope('concat-max-pools') as scope:
        filters_total_size = feature_size1 * len(filter_list)
        h_pools = tf.concat(3, poolings)
        h_pool_all = tf.reshape(h_pools, [-1, filters_total_size])

    ###dropout
    with tf.name_scope('dropout') as scope:
        keep_prob = tf.placeholder("float")
        h_pool1_drop = tf.nn.dropout(h_pool_all, keep_prob)

    ###readout Layer
    with tf.name_scope('full-connected') as scope:
        W_fc2 = cu.weight_variable([filters_total_size, label_size])
        b_fc2 = cu.bias_variable([label_size])
        h_pool1_flat = tf.reshape(h_pool1_drop, [-1, 1 * 1 * filters_total_size])
        y_conv = tf.nn.softmax(tf.matmul(h_pool1_flat, W_fc2) + b_fc2)

    ###train
    with tf.name_scope('compute-loss') as scope:
        cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    with tf.name_scope('train') as scope:
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    ###test
    with tf.name_scope("training-accuracy") as scope:
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy_summary = tf.scalar_summary("training accuracy", train_accuracy)
    with tf.name_scope("test-accuracy") as scope:
        correct_prediction_test = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        test_accuracy = tf.reduce_mean(tf.cast(correct_prediction_test, "float"))
        test_accuracy_summary = tf.scalar_summary("test accuracy", test_accuracy)

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())


    ###evaluate test data
    tf_saver = tf.train.Saver()
    tf_saver.restore(sess, model_name)

    test_accuracy_score = test_accuracy.eval(feed_dict={x: sentences_test, y_: labels_test, keep_prob: 1.0})
    print "test accuracy %g" % test_accuracy_score

    pred, ans = cu.evaluate_pr(tf, x, y_, sentences_test, labels_test, y_conv, keep_prob)

    #print pred
    #print ans

    #for i in range(len(pred)):
    #    print pred[i], ans[i], labels_test[i], raw_sen_test[i]

