import json
import string

import keras as K
import numpy as np

def loadData(max_words_number, max_review_len):
    print("Loading data, max unique words = %d words\n" % max_words_number)
    (train_x, train_y), (test_x, test_y) = \
        K.datasets.imdb.load_data(seed=1, num_words=max_words_number)
    train_x = K.preprocessing.sequence.pad_sequences(train_x, truncating='pre', padding='pre', maxlen=max_review_len)  # pad and chop!
    test_x = K.preprocessing.sequence.pad_sequences(test_x, truncating='pre', padding='pre', maxlen=max_review_len)

    return test_x, test_y, train_x, train_y


def split_review_data(reviews, sentiment_numerical_val, split=900, remove_punc=False, separation=" "):
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


def loadBayesData():
    reviews, sentiment_numerical_val = loadJSONData()
    training_set, training_labels, validation_set, validation_labels = split_review_data(reviews, sentiment_numerical_val)
    return training_set, training_labels, validation_set, validation_labels