"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(
    0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

from utilities.utilities import get_filenames_from_directory
from utilities.utilities import transform_data
from data_preprocessing.preprocessing_corpus import PreprocessingCorpus
from data_preprocessing.preprocessing_text import PreprocessingText
from data_preprocessing.train_test_split import TrainTestSplit
from lexicon_method.lexicon_method import LexiconMethod
from lexicon_method.textblob_de import TextBlobSentimentAnalysis
from multinomial_naive_bayes.multinomial_naive_bayes import MultinomialNaiveBayes
import pickle

path = "data/labeled_sentiment_data/pickle_files/"
preprocess_typ = "ekphrasis"

runprocessing = True
run_lexicon_method = True
run_textblob = True
run_mnb = True

############################################################################
# Preprocess Data
############################################################################
if(runprocessing):
    datafolders = [["labeled_sentiment_data/htw_data/join_data_clean.tsv", "\t", 0, 1]]
    data = PreprocessingCorpus(
        datafolders=datafolders, save_data_path=path, corpus_name="htw_")
    clean_data = data.ekphrasis_preprocessing()
    if(preprocess_typ == "stopwords"):
        clean_data = data.remove_stopwords(clean_data)
    if(preprocess_typ == "lemmatized"):
        clean_data = data.lemmatize_words(clean_data)
    data.save_clean_data(clean_data, path, preprocess_typ)

############################################################################
# Load Data
############################################################################
X_train = pickle.load(open("{}X_train_{}.pickle".format(path, preprocess_typ), "rb"))
y_train = pickle.load(open("{}y_train_{}.pickle".format(path, preprocess_typ), "rb"))
X_test = pickle.load(
    open("{}htw_X_clean_data_{}.pickle".format(path, preprocess_typ), "rb"))
y_test = pickle.load(
    open("{}htw_y_data.pickle".format(path), "rb"))

############################################################################
# Lexicon Method
############################################################################
if(run_lexicon_method):
    results_file_name = "{}/evaluation_htw_data_{}_lexiconmethod".format(preprocess_typ, preprocess_typ)

    print("run lexicon-method")
    sentiment_files = get_filenames_from_directory(
        'data/sentiment_lexicons/')
    lexicon_model = LexiconMethod(sentiment_files)
    lexicon_metric_list = lexicon_model.performance_analysis(
        transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, classification_report=True, save_pred=True)

############################################################################
# textblob-de
############################################################################
if(run_textblob):
    results_file_name = "{}/evaluation_htw_data_{}_textblob".format(
        preprocess_typ, preprocess_typ)

    print("run textblob-de")
    textblob_model = TextBlobSentimentAnalysis()
    metric_list = textblob_model.performance_analysis(
        transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, classification_report=True, save_pred=True)

############################################################################
# Multinomial Naive Bayes
############################################################################
if(run_mnb):
    results_file_name = "{}/evaluation_htw_data_{}_mnb".format(
        preprocess_typ, preprocess_typ)

    print("run multinomial naive bayes")
    mnb_model = MultinomialNaiveBayes(
        X_train, X_test, y_train, y_test, max_features=5000, min_df=2)
    mnb_model.encoding_textdata()
    model = mnb_model.fit_model()
    metric_list = mnb_model.performance_analysis(
        model, file_name=results_file_name, X_test=transform_data(X_test), verbose=True, accuracy=True, confusion_matrix=True, classification_report=True, save_pred=True)

