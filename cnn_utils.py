

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
