"""
Created by Florian Fricke.
"""

from utilities.utilities import get_filenames_from_directory
from utilities.utilities import transform_data
from data_preprocessing.preprocessing_corpus import PreprocessingCorpus
from data_preprocessing.preprocessing_text import PreprocessingText
from data_preprocessing.train_test_split import TrainTestSplit
from lexicon_method.lexicon_method import LexiconMethod
from lexicon_method.textblob_de import TextBlobSentimentAnalysis
from multinomial_naive_bayes.multinomial_naive_bayes import MultinomialNaiveBayes
import pickle

corpus_name = "sb10k_and_one_million_posts_corpus"  # scare
testing_corpus_name = "htw"  # sb10k_and_one_million_posts_corpus
path = "data/labeled_sentiment_data/pickle_files/"
preprocess_typ = "ekphrasis"

runprocessing = True
run_lexicon_method = True
run_textblob = True
run_mnb = True
run_single_predictions = True

############################################################################
# Preprocess Data
############################################################################
if(runprocessing):
    print("preprocess data")
    if(testing_corpus_name == "htw"):
        datafolders = [["labeled_sentiment_data/htw_data/join_data_clean.tsv", "\t", 0, 1]]
    if(testing_corpus_name == "sb10k_and_one_million_posts_corpus"):
        datafolders = [["labeled_sentiment_data/sb10k_corpus.tsv", "\t",
                        1, 4], ["labeled_sentiment_data/million_pos_corpus.tsv", "\t", 0, 1]]    
    if(testing_corpus_name == "scare"):
        datafolders = [["labeled_sentiment_data/scare_app_reviews.tsv", "\t", 0, 1]]

    data = PreprocessingCorpus(
        datafolders=datafolders, 
        save_data_path=path + (testing_corpus_name + "/" if testing_corpus_name != corpus_name else corpus_name + "/"))
    clean_data = data.ekphrasis_preprocessing()
    specific_stopwords = ["app", "wecker",
                          "google", "samsung", "apple", "android"]
    if(preprocess_typ == "stopwords"):
        clean_data = data.remove_stopwords(
            clean_data, specific_stopwords=specific_stopwords)
    if(preprocess_typ == "lemmatized"):
        clean_data = data.lemmatize_words(clean_data)
    data.save_clean_data(clean_data, path + 
                         (testing_corpus_name + "/" if testing_corpus_name != corpus_name else corpus_name + "/"), 
                         preprocess_typ)
    
    if(testing_corpus_name == corpus_name):
        split_data = TrainTestSplit(
            save_data_path=path + corpus_name + "/", preprocess_typ=preprocess_typ)
        X_train, X_test, y_train, y_test = split_data.split_to_train_test(
            test_size=0.3)

############################################################################
# Load Data
############################################################################
X_train = pickle.load(
    open("{}{}/X_train_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))
y_train = pickle.load(
    open("{}{}/y_train_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))

if(testing_corpus_name == corpus_name):
    X_test = pickle.load(
        open("{}{}/X_test_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))
    y_test = pickle.load(
        open("{}{}/y_test_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))
else:
    X_test = pickle.load(
        open("{}{}/X_clean_data_{}.pickle".format(path, testing_corpus_name, preprocess_typ), "rb"))
    y_test = pickle.load(
        open("{}{}/y_data.pickle".format(path, testing_corpus_name), "rb"))

############################################################################
# Single Predictions
############################################################################
if(run_single_predictions):
    print("Run Single Predition")
    text = ['Ich liebe euch und so!!!!', 'Ich hasse euch']
    pred = PreprocessingText(text)
    clean_data = pred.ekphrasis_preprocessing()
    if(preprocess_typ == "stopwords"):
        clean_data = pred.remove_stopwords(clean_data)
    if(preprocess_typ == "lemmatized"):
        clean_data = pred.lemmatize_words(clean_data)
    print(text)

############################################################################
# Lexicon Method
############################################################################
if(run_lexicon_method or run_single_predictions):
    results_file_name = "{}/{}/evaluation_{}_data_{}_lexiconmethod".format(
        preprocess_typ, corpus_name, testing_corpus_name, preprocess_typ)
    print("run lexicon-method")
    sentiment_files = get_filenames_from_directory(
        'data/sentiment_lexicons/')
    lexicon_model = LexiconMethod(sentiment_files)
    
    if(run_lexicon_method):
        lexicon_metric_list = lexicon_model.performance_analysis(
            transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, plotting_confusion_matrix=True, classification_report=True, save_pred=True)
   
    if(run_single_predictions):
        print("Lexicon Method: {}".format(lexicon_model.predict(transform_data(clean_data))))

############################################################################
# textblob-de
############################################################################
if(run_textblob or run_single_predictions):
    results_file_name = "{}/{}/evaluation_{}_data_{}_textblob".format(
        preprocess_typ, corpus_name, testing_corpus_name, preprocess_typ)
    print("run textblob-de")
    textblob_model = TextBlobSentimentAnalysis()
    
    if(run_textblob):
        metric_list = textblob_model.performance_analysis(
            transform_data(X_test), y_test, file_name=results_file_name, verbose=True, confusion_matrix=True, plotting_confusion_matrix=True, classification_report=True, save_pred=True)

    if(run_single_predictions):
        print("textblob-de: {}".format(textblob_model.predict(transform_data(clean_data))))

############################################################################
# Multinomial Naive Bayes
############################################################################
if(run_mnb or run_single_predictions):
    results_file_name = "{}/{}/evaluation_{}_data_{}_mnb".format(
        preprocess_typ, corpus_name, testing_corpus_name, preprocess_typ)

    print("run multinomial naive bayes")
    mnb_model = MultinomialNaiveBayes(
        X_train, X_test, y_train, y_test, max_features=5000, min_df=2)
    mnb_model.encoding_textdata()
    model = mnb_model.fit_model()
    
    if(run_mnb):
        metric_list = mnb_model.performance_analysis(
            model, file_name=results_file_name, X_test=transform_data(X_test), verbose=True, accuracy=True, confusion_matrix=True, plotting_confusion_matrix=True, classification_report=True, save_pred=False)
    if(run_single_predictions):
        print("Multinomial Naive Bayes: {}".format(
        mnb_model.predict(model, transform_data(clean_data))))
