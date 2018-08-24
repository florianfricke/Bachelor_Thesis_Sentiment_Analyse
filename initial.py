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

runprocessing = False
run_lexicon_method = True
run_textblob = True
run_mnb = True
run_single_predictions = True

############################################################################
# Preprocess Data
############################################################################
if(runprocessing):
    datafolders = [["labeled_sentiment_data/sb10k_corpus.tsv", "\t",
                     1, 4], ["labeled_sentiment_data/million_pos_corpus.tsv", "\t", 0, 1]]
    data = PreprocessingCorpus(datafolders=datafolders, save_data_path=path)
    clean_data = data.ekphrasis_preprocessing()
    if(preprocess_typ == "stopwords"):
        clean_data = data.remove_stopwords(clean_data)
    data.save_clean_data(clean_data, path, preprocess_typ)

    split_data = TrainTestSplit(save_data_path=path, preprocess_typ=preprocess_typ)
    X_train, X_test, y_train, y_test = split_data.split_to_train_test(
        test_size=0.3)

############################################################################
# Load Data
############################################################################
X_train = pickle.load(open("{}X_train_{}.pickle".format(path, preprocess_typ), "rb"))
y_train = pickle.load(open("{}y_train_{}.pickle".format(path, preprocess_typ), "rb"))
X_test = pickle.load(open("{}X_test_{}.pickle".format(path, preprocess_typ), "rb"))
y_test = pickle.load(open("{}y_test_{}.pickle".format(path, preprocess_typ), "rb"))

############################################################################
# Single Predictions
############################################################################
if(run_single_predictions):
    print("Run Single Predition")
    text = ['Ich liebe euch', 'Ich hasse euch']
    pred = PreprocessingText(text)
    clean_data = pred.ekphrasis_preprocessing()
    if(preprocess_typ == "stopwords"):
        clean_data = pred.remove_stopwords(clean_data)
    print(text)

############################################################################
# Lexicon Method
############################################################################
if(run_lexicon_method or run_single_predictions):
    results_file_name = "{}/evaluation_{}_lexiconmethod".format(
        preprocess_typ, preprocess_typ)
    print("run lexicon-method")
    sentiment_files = get_filenames_from_directory(
        'data/sentiment_lexicons/')
    lexicon_model = LexiconMethod(sentiment_files)
    
    if(run_lexicon_method):
        lexicon_metric_list = lexicon_model.performance_analysis(
            transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, classification_report=True, save_pred=True)
    
    if(run_single_predictions):
        print("Lexicon Method: {}".format(lexicon_model.predict(text)))

############################################################################
# textblob-de
############################################################################
if(run_textblob or run_single_predictions):
    results_file_name = "{}/evaluation_{}_textblob".format(
        preprocess_typ, preprocess_typ)
    print("run textblob-de")
    textblob_model = TextBlobSentimentAnalysis()

    if(run_textblob):
        metric_list = textblob_model.performance_analysis(
            transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, classification_report=True, save_pred=True)

    if(run_single_predictions):
        print("textblob-de: {}".format(textblob_model.predict(text)))

############################################################################
# Multinomial Naive Bayes
############################################################################
if(run_mnb or run_single_predictions):
    results_file_name = "{}/evaluation_{}_mnb".format(
        preprocess_typ, preprocess_typ)
    print("run multinomial naive bayes")
    mnb_model = MultinomialNaiveBayes(
        X_train, X_test, y_train, y_test, max_features=5000, min_df=2)
    mnb_model.encoding_textdata()
    model = mnb_model.fit_model()

    if(run_mnb):
        metric_list = mnb_model.performance_analysis(
            model, file_name=results_file_name, X_test=transform_data(X_test), verbose=True, accuracy=True, confusion_matrix=True, classification_report=True, save_pred=True)
    
    if(run_single_predictions):
        print("Multinomial Naive Bayes: {}".format(mnb_model.predict(model, text)))
