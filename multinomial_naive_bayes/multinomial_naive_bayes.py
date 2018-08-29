"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(
    0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from utilities.utilities import transform_data
from utilities.utilities import plot_confusion_matrix
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class MultinomialNaiveBayes:
    def __init__(self, X_train, X_test, y_train, y_test, **kwargs):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = CountVectorizer(lowercase=False, max_features=kwargs.get('max_features', None), min_df=kwargs.get('min_df', 0))
      
    def encoding_textdata(self):
        self.X_train = self.cv.fit_transform(
            pd.Series(transform_data(self.X_train)))
        self.X_test = self.cv.transform(pd.Series(transform_data(self.X_test)))
        print("Dimension X_train Bag of Words: {}".format(self.X_train.shape))
        print("Dimension X_test Bag of Words: {}".format(self.X_test.shape))
        return None

    def fit_model(self):
        model = MultinomialNB()
        model.fit(self.X_train, self.y_train)
        return model        

    def performance_analysis(self, model, X_test=[], file_name="", verbose=True, accuracy=True, confusion_matrix=True, plotting_confusion_matrix=True, classification_report=True, save_pred=True):
        y_pred = model.predict(self.X_test)
        metric_list = list()
        
        with open('results/{}.txt'.format(file_name), 'w', encoding="utf-8") as f:
            if(accuracy):
                accuracy = model.score(self.X_test, self.y_test)
                metric_list.append(accuracy)
                if(verbose):
                    print("model accuracy:", file=f)
                    print(accuracy, file=f)
            
            if(confusion_matrix):
                confusion_matrix = metrics.confusion_matrix(self.y_test, y_pred, labels=[
                    "positive", "neutral", "negative"])
                metric_list.append(confusion_matrix)
                
                if(verbose):
                    print("confusion matrix:", file=f)
                    print(confusion_matrix, file=f)

                if(plotting_confusion_matrix):
                    from matplotlib import rcParams
                    rcParams.update({'figure.autolayout': True})

                    # plot confusion matrix
                    plt.figure(dpi=600)
                    plot_confusion_matrix(confusion_matrix, classes=['positiv', 'neutral', 'negativ'],
                                        title='Wahrheitsmatrix')
                    plt.savefig(
                        'results/{}_confusion_matrix_1.png'.format(file_name))
                    
                    # plot normalized confusion matrix
                    plt.figure(dpi=600)
                    plot_confusion_matrix(confusion_matrix, classes=['positiv', 'neutral', 'negativ'], normalize=True,
                                        title='normalisierte Wahrheitsmatrix')
                    plt.savefig(
                        'results/{}_confusion_matrix_2.png'.format(file_name))

            if(classification_report):
                classification_report = metrics.classification_report(self.y_test, y_pred)
                metric_list.append(classification_report)
                if(verbose):
                    print("classification report:", file=f)
                    print(classification_report, file=f)
        
        if(save_pred):
            with open('results/predictions/{}_predictions.txt'.format(file_name), 'w', encoding="utf-8") as f:
                for i in range(len(y_pred)):
                    print("{}\t{}".format(
                        y_pred[i], X_test[i]), file=f)
        
        return metric_list

    def predict(self, model, testdata):
        x = self.cv.transform(pd.Series(testdata))
        return model.predict(x)
