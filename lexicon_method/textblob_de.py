"""
Created by Florian Fricke.
"""

from textblob_de import TextBlobDE as TextBlob
from sklearn import metrics
from tqdm import tqdm
from utilities.utilities import transform_data
from utilities.utilities import plot_confusion_matrix
import pickle
import matplotlib.pyplot as plt

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

    def performance_analysis(self, X_test, y_test, file_name="", verbose=True, confusion_matrix=True, plotting_confusion_matrix=True, classification_report=True, save_pred=True):
        y_pred = self.predict(X_test)
        metric_list = list()

        with open('results/{}.txt'.format(file_name), 'w', encoding="utf-8") as f:
            if(confusion_matrix):
                confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=[
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
                classification_report = metrics.classification_report(
                    y_test, y_pred)
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
