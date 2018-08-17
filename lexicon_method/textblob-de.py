"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

from textblob_de import TextBlobDE as TextBlob
from sklearn import metrics
from tqdm import tqdm
from utilities.utilities import transform_data
import pickle

class TextBlobSentimentAnalysis:    
    def predict(self, textarray):
        prediction = list()
        for text in tqdm(textarray):
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity

            if(sentiment == 0):
                prediction.append("neutral")
            if(sentiment > 0):
                prediction.append("positive")
            if(sentiment < 0):
                prediction.append("negative")
        return prediction

    def performance_analysis(self, X_test, y_test, verbose=True, classification_report=True, confusion_matrix=True):
        y_pred = self.predict(X_test)
        metric_list = list()

        if(confusion_matrix):
            confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[
                "positive", "neutral", "negative"])
            metric_list.append(confusion_matrix)
            if(verbose):
                print("confusion matrix:")
                print(confusion_matrix)

        if(classification_report):
            classification_report = metrics.classification_report(
                y_test, y_pred)
            metric_list.append(classification_report)
            if(verbose):
                print("classification report:")
                print(classification_report)
        return metric_list

if __name__ == '__main__':
    model = TextBlobSentimentAnalysis()

    path = "data/labeled_sentiment_data/pickle_files/"
    preprocess_typ = "stopwords"
    X_test = pickle.load(open(
        path + "X_test_" + preprocess_typ + ".pickle", "rb"))

    y_test = pickle.load(open(
        path + "y_test_" + preprocess_typ + ".pickle", "rb"))

    X_test = transform_data(X_test)

    metric_list = model.performance_analysis(
        X_test, y_test, verbose=True, classification_report=True, confusion_matrix=True)

    text = ['Ich liebe euch', 'Ich hasse euch']
    result = model.predict(text)
    print(result)
