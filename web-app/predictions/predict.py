"""
Created by Florian Fricke.
"""
import sys
import os
sys.path.insert(
    0, "{}/datastories_semeval2017_task4".format(os.getcwd()))
from utilities_nn.data_loader import get_embeddings, Task4Loader
sys.path.insert(
    0, "{}".format(os.getcwd()))
from utilities.utilities import get_filenames_from_directory
from utilities.utilities import transform_data
from data_preprocessing.preprocessing_text import PreprocessingText
from lexicon_method.lexicon_method import LexiconMethod
from lexicon_method.textblob_de import TextBlobSentimentAnalysis
from multinomial_naive_bayes.multinomial_naive_bayes import MultinomialNaiveBayes
from keras.models import model_from_json

import keras.models
import pickle


def predict_sentiment(text, preprocess_typ, nn_model1, nn_model2, word_indices):
    result_data = []

    ############################################################################
    # Load Data
    ############################################################################
    path = "data/labeled_sentiment_data/pickle_files/"
    corpus_name = "mixed_corpus_1"

    X_train = pickle.load(
        open("{}{}/X_train_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))
    y_train = pickle.load(
        open("{}{}/y_train_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))
    X_test = pickle.load(
        open("{}{}/X_test_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))
    y_test = pickle.load(
        open("{}{}/y_test_{}.pickle".format(path, corpus_name, preprocess_typ), "rb"))

    ############################################################################
    # Single Predictions
    ############################################################################
    # text = ['Ich liebe euch und so!!!! Es gefällt mir hier.']
    # text = ['Alles Mist hier!!!! Ich hasse euch']
    pred = PreprocessingText(text)
    clean_data = pred.ekphrasis_preprocessing()
    if(preprocess_typ == "stopwords"):
        clean_data = pred.remove_stopwords(clean_data)
    if(preprocess_typ == "lemmatized"):
        clean_data = pred.lemmatize_words(clean_data)

    ############################################################################
    # Lexicon Method
    ############################################################################
    sentiment_files = get_filenames_from_directory(
        'data/sentiment_lexicons/')
    lexicon_model = LexiconMethod(sentiment_files)
    result_data.append(lexicon_model.predict(transform_data(clean_data)))

    ############################################################################
    # textblob-de
    ############################################################################
    textblob_model = TextBlobSentimentAnalysis()
    result_data.append(textblob_model.predict(transform_data(clean_data)))

    ############################################################################
    # Multinomial Naive Bayes
    ############################################################################
    mnb_model = MultinomialNaiveBayes(
        X_train, X_test, y_train, y_test, max_features=5000, min_df=2)
    mnb_model.encoding_textdata()
    model = mnb_model.fit_model()

    result_data.append(mnb_model.predict(model, transform_data(clean_data)))
    
    ############################################################################
    # Artificial Neural Network
    ############################################################################
    print("decode data to word vectors")
    data = []
    data.append(clean_data)
    loader = Task4Loader(word_indices, text_lengths=50, loading_data=False,
                        preprocess_typ=preprocess_typ)
                
    decode_data = loader.decode_data_to_embeddings(
        clean_data, ['positive','negative','neutral'])  # decode data to word vectors
    pred1 = nn_model1.predict(decode_data[0])
    pred1 = pred1.argmax(axis=1)
    result_data.append(pred1)

    pred2 = nn_model1.predict(decode_data[0])
    pred2 = pred2.argmax(axis=1)
    result_data.append(pred2)

    return [result_data[i][0] for i in range(len(result_data))]
