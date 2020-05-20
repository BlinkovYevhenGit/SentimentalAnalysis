import json
import string

import tensorflow.keras as K
import numpy as np

def loadData(top_words, max_review_len):
    print("Loading data, max unique words = %d words\n" % top_words)
    (train_x, train_y), (test_x, test_y) = \
        K.datasets.imdb.load_data(seed=1,num_words=top_words)
    train_x = K.preprocessing.sequence.pad_sequences(train_x, truncating='pre', padding='pre', maxlen=max_review_len)  # pad and chop!
    test_x = K.preprocessing.sequence.pad_sequences(test_x, truncating='pre', padding='pre', maxlen=max_review_len)

    return test_x, test_y, train_x, train_y


def split_review_data(reviews, sentiment_numerical_val, split=900, remove_punc=True, separation=" "):
    training_set = []
    training_labels = []
    validation_set = []
    validation_labels = []

    for i, r in enumerate(reviews):
        # if i==0: print(str(r['content'])); print(dict(r).keys())
        cv = int(r["cv"])
        sent = sentiment_numerical_val[r["sentiment"]]
        content_string = ""
        for sentence in r["content"]:
            for word in sentence:
                content_string += word[0].lower() + separation

        if remove_punc:
            exclude = set(string.punctuation)
            content_string = ''.join(character for character in content_string if character not in exclude)

        if 0 < cv < split:
            training_set.append(content_string)
            training_labels.append(sent)
        else:
            validation_set.append(content_string)
            validation_labels.append(sent)

    return training_set, np.array(training_labels), validation_set, np.array(validation_labels)


def loadJSONData():
    with open("reviews.json", mode="r", encoding="utf-8") as f:
        reviews = json.load(f)
    sentiment_numerical_val = {
        'NEG': 0,
        'POS': 1
    }
    return reviews, sentiment_numerical_val
def getIMDB_dataForBayes(top_words, maxlen):
    #(train_data_raw, train_labels), (test_data_raw, test_labels) = K.datasets.imdb.load_data()#num_words=50000, maxlen=5000
    train_data_raw, train_labels, test_data_raw, test_labels= loadData(top_words, maxlen)
    words2idx = K.datasets.imdb.get_word_index()
    idx2words = {idx: word for word, idx in words2idx.items()}
    train_set = []
    # maxLenTest = 0
    # maxLenTrain = 0

    for review in train_data_raw:
        train_ex = []
        length = 0
        for x in review[0:]:
            # length = length + 1
            if x > 3:
                word = idx2words[x - 3]
                train_ex.append(word)
        train_ex = ' '.join(train_ex)
        # if length > maxLenTrain:
        #     maxLenTrain = length
        train_set.append(train_ex)
    test_set = []
    for review in test_data_raw:
        length = 0
        test_ex = []
        for x in review[0:]:
            # length = length + 1
            if x > 3:
                word = idx2words[x - 3]
                test_ex.append(word)
        # if length > maxLenTest:
        #     maxLenTest = length
        test_ex = ' '.join(test_ex)
        test_set.append(test_ex)
    return train_set,train_labels,test_set,test_labels


def loadBayesData(top_words, max_words_number):
    #reviews, sentiment_numerical_val = loadJSONData()
    training_set, training_labels, validation_set, validation_labels = getIMDB_dataForBayes(top_words=top_words, maxlen=max_words_number)
    return training_set, training_labels, validation_set, validation_labels