"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(
    0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

import pickle
from sklearn.model_selection import train_test_split

class TrainTestSplit:
    def __init__(self, X_pickle, y_pickle):
        self.X = pickle.load(open(X_pickle, "rb"))
        self.y = pickle.load(open(y_pickle, "rb"))

    def split_to_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                        test_size=0.3,
                                                        stratify=self.y,
                                                        random_state=27,
                                                        shuffle=True)
        print("training set: {} and test set: {}".format(
            len(X_train), len(X_test)))
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    path = "data/labeled_sentiment_data/pickle_files/"
    data = TrainTestSplit(
        path + "X_clean_data_stopwords.pickle", path + "y_data.pickle")
    X_train, X_test, y_train, y_test = data.split_to_train_test()

    pickle.dump(X_train, open(
        path + "X_train_stopwords.pickle", "wb"))
    pickle.dump(X_test, open(
        path + "X_test_stopwords.pickle", "wb"))
    pickle.dump(y_train, open(
        path + "y_train_stopwords.pickle", "wb"))
    pickle.dump(y_test, open(
        path + "y_test_stopwords.pickle", "wb"))

