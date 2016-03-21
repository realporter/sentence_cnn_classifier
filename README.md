# sentence_cnn_classifier
A sentence classifier using Convolutional Neural Networks in TensorFlow

### Description


### Data Flow
#### Generate feature vector for each sentence
generate_sentence_vectors.py

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


#### Test the classifier


