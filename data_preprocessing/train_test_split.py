"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(
    0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

import pickle
from sklearn.model_selection import train_test_split
from utilities.utilities import save_data

class TrainTestSplit:
    def __init__(self, save_data_path, preprocess_typ, **kwargs):
        self.corpus_name = kwargs.get('corpus_name', "")
        self.X = pickle.load(open("{}{}X_clean_data_{}.pickle".format(save_data_path, self.corpus_name, preprocess_typ), "rb"))
        self.y = pickle.load(open("{}{}y_data.pickle".format(save_data_path, self.corpus_name), "rb"))
        self.save_data_path = save_data_path
        self.preprocess_typ = preprocess_typ

    def split_to_train_test(self, test_size=0.3):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                        test_size=test_size,
                                                        stratify=self.y,
                                                        random_state=27,
                                                        shuffle=True)
        print("training set: {} and test set: {}".format(
            len(X_train), len(X_test)))
        
        save_data(X_train, path=self.save_data_path,
                  filename="{}X_train_{}".format(self.corpus_name, self.preprocess_typ))
        save_data(X_test, path=self.save_data_path,
                  filename="{}X_test_{}".format(self.corpus_name, self.preprocess_typ))
        save_data(y_train, path=self.save_data_path,
                  filename="{}y_train_{}".format(self.corpus_name, self.preprocess_typ))
        save_data(y_test, path=self.save_data_path,
                  filename="{}y_test_{}".format(self.corpus_name, self.preprocess_typ))
        return X_train, X_test, y_train, y_test

