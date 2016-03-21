import numpy as np
import tensorflow as tf
import random

def pad_sentence(sentences, w2v_size, max_length):
    #pad_init_value = np.random.normal(0, 0.1, w2v_size).tolist()
    pad_init_value = [0.00001] * w2v_size
    for i in range(len(sentences)):
        pad_length = max_length - len(sentences[i])
        pad_list = [pad_init_value] * pad_length
        sentences[i] = sentences[i] + pad_list

def change_label_type(original_label):
    label = []
    for i in range(len(original_label)):
        if original_label[i] == 1:
            label.append([1, 0])
        else:
            label.append([0, 1])
    return label

def build_data(pos_fmatrix, neg_fmatrix, pos_label, neg_label, pos_sen, neg_sen):
    fmatrix, original_label, sen = pos_fmatrix + neg_fmatrix, pos_label + neg_label, pos_sen + neg_sen
    label = change_label_type(original_label)
    return fmatrix, label, sen

def build_and_shuffle_data(pos_fmatrix, neg_fmatrix, pos_label, neg_label, pos_sen, neg_sen):
    data = np.array([pos_fmatrix + neg_fmatrix, pos_label + neg_label, pos_sen + neg_sen])
    shuffled_indices = np.random.permutation(np.arange(len(data[0])))
    ###suffile data
    fmatrix = data[0][shuffled_indices].tolist()
    original_label = data[1][shuffled_indices].tolist()
    sen = data[2][shuffled_indices].tolist()
    label = change_label_type(original_label)
    return fmatrix, label, sen

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_Nx1(x, length):
    return tf.nn.max_pool(x, ksize=[1, length, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

def get_batch_rand(sentences_train, labels_train, batch_size):
    sentence_batch = []
    label_batch = []
    for i in range(batch_size):
        idx = random.randint(0, len(sentences_train) - 1)
        sentence_batch.append(sentences_train[idx])
        label_batch.append(labels_train[idx])
    return sentence_batch, label_batch

def get_batch_balanced(pos_fmatrix, neg_fmatrix, pos_label, neg_label, batch_size):
    sentence_batch = []
    label_batch = []
    limit = batch_size / 2
    for i in range(limit):
        idx = random.randint(0, len(pos_fmatrix) - 1)
        sentence_batch.append(pos_fmatrix[idx])
        label_batch.append(pos_label[idx])
    for i in range(limit):
        idx = random.randint(0, len(neg_fmatrix) - 1)
        sentence_batch.append(neg_fmatrix[idx])
        label_batch.append(neg_label[idx])
    return sentence_batch, label_batch

def evaluate_pr(tf, x, y_, sentences_test, labels_test, y_conv, keep_prob):

    pred, ans = tf.argmax(y_conv,1), tf.argmax(y_, 1)
    pred = pred.eval(feed_dict={x: sentences_test, y_: labels_test, keep_prob: 1.0})
    ans = ans.eval(feed_dict={x: sentences_test, y_: labels_test, keep_prob: 1.0})

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(ans)):
        if pred[i] == 0:
            if pred[i] == ans[i]:
                tp += 1
            else:
                fp += 1
        else:
            if pred[i] == ans[i]:
                tn += 1
            else:
                fn += 1

    precision = 0
    recall = 0
    if (tp + fp) > 0:
        precision = float(tp) / (tp + fp)
    if (tp + fn) > 0:
        recall = float(tp) / (tp + fn)

    print 'tp: %d, tn: %d, fp: %d, fn: %d' % (tp, tn, fp, fn)
    print 'precision: ', precision
    print 'recall:    ', recall

    return pred, ans
