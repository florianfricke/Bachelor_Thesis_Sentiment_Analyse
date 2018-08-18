"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

from sklearn import metrics
from os.path import join
from tqdm import tqdm
from utilities.utilities import get_filenames_from_directory
from utilities.utilities import transform_data
import pickle

class LexiconMethod:

    def __init__(self, sentiment_files):
        self.seperator = "\t"
        self.negative_words = self.read_lexicon(sentiment_files[0])
        self.neutral_words = self.read_lexicon(sentiment_files[1])
        self.positive_words = self.read_lexicon(sentiment_files[2])
        
        print("negative words: {}, neutral words: {}, positive words: {}".format(
            len(self.negative_words), len(self.neutral_words), len(self.positive_words)))

    def read_lexicon(self, lexicon_file, path="data/sentiment_lexicons/"):
        sentiment_words = []
    
        for line_id, line in enumerate(open(join(path, lexicon_file), "r", encoding="utf-8").readlines()):
            try:
                columns = line.rstrip().split(self.seperator)
                sentiment_words.append(columns[0])
            except Exception:
                print("\nWrong format in line:{} in file:{}".format(
                    line_id, lexicon_file))
                raise Exception
        return sentiment_words

    def get_sentiment(self, word):
        if word in self.neutral_words:
            return 0
        elif word in self.positive_words:
            return 1
        elif word in self.negative_words:
            return -1
        else:
            # print("{} - don't exist in sentiment lexicon".format(word))
            return 0

    def predict(self, textarray):
        prediction = list()
        for text in tqdm(textarray):
            tokens = text.split()
            summe = sum([self.get_sentiment(t) for t in tokens])

            if(summe == 0):
                prediction.append("neutral")
            if(summe > 0):
                prediction.append("positive")
            if(summe < 0):
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
                classification_report = metrics.classification_report(y_test, y_pred)
                metric_list.append(classification_report)
                if(verbose):
                    print("classification report:", file=f)
                    print(classification_report, file=f)
        return metric_list

