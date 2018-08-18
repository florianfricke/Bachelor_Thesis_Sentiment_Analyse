"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(
    0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

from utilities.utilities import get_filenames_from_directory
from utilities.utilities import transform_data
from data_preprocessing.preprocessing import Preprocessing
from data_preprocessing.train_test_split import TrainTestSplit
from lexicon_method.lexicon_method import LexiconMethod
from lexicon_method.textblob_de import TextBlobSentimentAnalysis
from multinomial_naive_bayes.multinomial_naive_bayes import MultinomialNaiveBayes
import pickle

path = "data/labeled_sentiment_data/pickle_files/"
preprocess_typ = "emphrasis"

############################################################################
# Preprocess Data
############################################################################
runprocessing = False

if(runprocessing):
    data = Preprocessing()
    clean_data = data.emphrasis_preprocessing()
    data.save_clean_data(clean_data, path, preprocess_typ)
    # clean_data = data.remove_stopwords(clean_data)

############################################################################
# Load Data
############################################################################
X_train = pickle.load(open("{}X_train_{}.pickle".format(path, preprocess_typ), "rb"))
y_train = pickle.load(open("{}y_train_{}.pickle".format(path, preprocess_typ), "rb"))
X_test = pickle.load(open("{}X_test_{}.pickle".format(path, preprocess_typ), "rb"))
y_test = pickle.load(open("{}y_test_{}.pickle".format(path, preprocess_typ), "rb"))

############################################################################
# Lexicon Method
############################################################################
run_lexicon_method = True

if(run_lexicon_method):
    results_file_name = "{}_lexiconmethod".format(preprocess_typ)

    print("run lexicon-method")
    sentiment_files = get_filenames_from_directory(
        'data/sentiment_lexicons/')
    lexicon_model = LexiconMethod(sentiment_files)
    lexicon_metric_list = lexicon_model.performance_analysis(
        transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, classification_report=True)
    text = ['Ich liebe euch', 'Ich hasse euch']
    result = lexicon_model.predict(text)
    print(result)

############################################################################
# textblob-de
############################################################################
run_textblob = False

if(run_textblob):
    results_file_name = "{}_textblob".format(preprocess_typ)

    print("run textblob-de")
    textblob_model = TextBlobSentimentAnalysis()
    metric_list = textblob_model.performance_analysis(
        transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, classification_report=True)
    text = ['Ich liebe euch', 'Ich hasse euch']
    result = textblob_model.predict(text)
    print(result)

############################################################################
# Multinomial Naive Bayes
############################################################################
run_mnb = True

if(run_mnb):
    results_file_name = "{}_mnb".format(preprocess_typ)

    print("run multinomial naive bayes")
    mnb_model = MultinomialNaiveBayes(
        X_train, X_test, y_train, y_test, max_features=5000, min_df=2)
    mnb_model.encoding_textdata()
    model = mnb_model.fit_model()
    metric_list = mnb_model.performance_analysis(
        model, file_name=results_file_name, verbose=True, accuracy=True, confusion_matrix=True, classification_report=True)
    print(mnb_model.predict(model, ["Ich liebe euch", "Ich hasse euch"]))
