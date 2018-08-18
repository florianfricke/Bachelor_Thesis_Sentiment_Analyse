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

    def performance_analysis(self, X_test, y_test, file_name="", verbose=True, confusion_matrix=True, classification_report=True):
        y_pred = self.predict(X_test)
        metric_list = list()

        with open('results/evaluation_{}.txt'.format(file_name), 'w') as f:
            if(confusion_matrix):
                confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[
                    "positive", "neutral", "negative"])
                metric_list.append(confusion_matrix)
                if(verbose):
                    print("confusion matrix:", file=f)
                    print(confusion_matrix, file=f)

            if(classification_report):
                classification_report = metrics.classification_report(
                    y_test, y_pred)
                metric_list.append(classification_report)
                if(verbose):
                    print("classification report:", file=f)
                    print(classification_report, file=f)
        return metric_list
