# sentence_cnn_classifier
A sentence classifier using Convolutional Neural Networks in TensorFlow

### Description

### Data
http://www.cs.cornell.edu/people/pabo/movie-review-data/


### Data Flow
#### Generate feature vector for each sentence
If you want to use a pre-trained word embedding, check the GoogleNews-vectors-negative300.bin.gz from
Google word2vec project https://code.google.com/archive/p/word2vec/ 


Use **generate_sentence_vectors.py** to convert raw sentences into feature vector.

    usage: generate_sentence_vectors.py [-h] -w2v W2V [-w2v_size W2V_SIZE] -s
                                    SENTENCES -l {0,1} -v SENTENCE_VECTORS
    Generate feature vector for sentences
    optional arguments:
      -h, --help            show this help message and exit
      -w2v W2V              word2vector Binary File
      -w2v_size W2V_SIZE    word2vector size (default: 300)
      -s SENTENCES, --sentences SENTENCES
                            input sentence file
      -l {0,1}, --label {0,1}
                            label of the data, [0: negative, 1: positive]
      -v SENTENCE_VECTORS, --sentence_vectors SENTENCE_VECTORS
                            output sentence vectors

#### Train sentence CNN classifier


    usage: sentence_cnn_train.py [-h] -pf POS_FILE -nf NEG_FILE
                                 [-w2v_size W2V_SIZE] [--label_size LABEL_SIZE]
                                 [-b TRAINING_BATCH_SIZE] -m MODEL_OUTPUT
                                 [-test_size TEST_SIZE] [-iterations ITERATIONS]
                                 [-dropout_rate DROPOUT_RATE]
                                 [-e EVALUATE_PER_N_BATCHES]

    Generate feature vector for sentences

    optional arguments:
      -h, --help            show this help message and exit
      -pf POS_FILE, --pos_file POS_FILE
                            positive feature vectors for training
      -nf NEG_FILE, --neg_file NEG_FILE
                            negitive feature vectors for training
      -w2v_size W2V_SIZE    word2vector size (default: 300)
      --label_size LABEL_SIZE
                            how many classes? (default: 2)
      -b TRAINING_BATCH_SIZE, --training_batch_size TRAINING_BATCH_SIZE
                            size of each batch when training (default: 50)
      -m MODEL_OUTPUT, --model_output MODEL_OUTPUT
                            the trained model
      -test_size TEST_SIZE  test data size for each class
      -iterations ITERATIONS
                            number of training iterations (default: 1000)
      -dropout_rate DROPOUT_RATE
                            droupout rate (default: 0.5)
      -e EVALUATE_PER_N_BATCHES, --evaluate_per_n_batches EVALUATE_PER_N_BATCHES
                            evaluate training and test accuracy per N batches
                            (default: 50)


#### Test the classifier


