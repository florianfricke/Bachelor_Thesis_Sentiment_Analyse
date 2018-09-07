"""
Created by Florian Fricke.
"""

from sklearn import metrics
from keras.models import load_model
import sys
import pickle
import os
import matplotlib.pyplot as plt
import itertools
import numpy as np

def performance_analysis(testing, model, file_name="", file_information="", verbose=True, accuracy=True, confusion_matrix=True, plotting_confusion_matrix=True, classification_report=True, **kwargs):
    with open('results_artificial_neural_network/{}.txt'.format(file_name), 'w') as f:
        print(file_information, file=f)
        y_pred = model.predict(testing[0])
        y_pred = y_pred.argmax(axis=1)
        metric_list = list()
        if(accuracy):
            accuracy = model.evaluate(testing[0], testing[1])
            metric_list.append(accuracy)
            if(verbose):
                print("\nmodel accuracy (loss value, accuracy):", file=f)
                print(accuracy, file=f)

        if(confusion_matrix):
            confusion_matrix = metrics.confusion_matrix(testing[1].argmax(
                axis=1), y_pred)
            metric_list.append(confusion_matrix)
            if(verbose):
                print("\nconfusion matrix:", file=f)
                print(confusion_matrix, file=f)

            if(plotting_confusion_matrix):
                import matplotlib as mpl
                mpl.rcParams.update(mpl.rcParamsDefault)
                from matplotlib import rcParams
                rcParams.update({'figure.autolayout': True})

                # Plot non-normalized confusion matrix
                plt.figure(dpi=600)
                plot_confusion_matrix(confusion_matrix, classes=['positiv', 'neutral', 'negativ'],
                                        title='Wahrheitsmatrix')
                plt.savefig(
                    'results_artificial_neural_network/{}_confusion_matrix_1.png'.format(file_name))

                # Plot normalized confusion matrix
                plt.figure(dpi=600)
                plot_confusion_matrix(confusion_matrix, classes=['positiv', 'neutral', 'negativ'], normalize=True,
                                        title='normalisierte Wahrheitsmatrix')
                plt.savefig(
                    'results_artificial_neural_network/{}_confusion_matrix_2.png'.format(file_name))


        if(classification_report):
            classification_report = metrics.classification_report(
                testing[1].argmax(axis=1), y_pred)
            metric_list.append(classification_report)
            if(verbose):
                print("\nclassification report:", file=f)
                print(classification_report, file=f)

        if(kwargs.get('save_pred', False) and kwargs.get('X_test_unprocessed', False)):
            with open('results_artificial_neural_network/predictions/{}_predictions.txt'.format(file_name), 'w', encoding="utf-8") as f:
                for i in range(len(y_pred)):
                    if y_pred[i] == 0:
                        pred = "negative"
                    elif y_pred[i] == 1:
                        pred = "neutral"
                    elif y_pred[i] == 2:
                        pred = "positive"
                    else:
                        pred=""
                    print("{}\t{}".format(
                        pred, X_test_unprocessed[i]), file=f)
    return metric_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Wahrheitsmatrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')






############################################################################
# Evaluate Data
############################################################################
corpus_name = "htw"
pickle_path = "data/labeled_sentiment_data/pickle_files/{}/".format(
    corpus_name)
preprocess_typ = "ekphrasis"
model_file_number = 4
file_information = ""
attention_mechanism = True
corpusname = "sb10k_and_one_million_posts_corpus"

print("load model_{}_{}".format(preprocess_typ, model_file_number))
if(attention_mechanism):
    from kutilities.layers import Attention
    nn_model = load_model(
        'results_artificial_neural_network/{}/{}/model_{}_{}.hdf5'.format(
            preprocess_typ, corpusname, preprocess_typ, model_file_number),
        custom_objects={'Attention': Attention})
else:
    nn_model = load_model(
        'results_artificial_neural_network/{}/{}/model_{}_{}.hdf5'.format(
            preprocess_typ, corpusname, preprocess_typ, model_file_number))

#___________________Evaluate sb10k + One Million Posts Korpus___________________
if(False):
    file_name = "{}/{}/evaluation_scare_{}_{}".format(preprocess_typ, corpusname,
                                                   preprocess_typ, model_file_number)
    X_test_unprocessed = pickle.load(
        open("{}X_data_unprocessed.pickle".format(pickle_path), "rb"))
    testing_data = pickle.load(
        open("{}testing_data_nn_{}.pickle".format(pickle_path, preprocess_typ), "rb"))

    performance_analysis(testing_data, nn_model, file_name=file_name, file_information=file_information, verbose=True, accuracy=True,
                         confusion_matrix=True, classification_report=True, save_pred=True, X_test_unprocessed=X_test_unprocessed)

#_______________________________Evaluate HTW Data_______________________________
if(True):
    file_name = "{}/{}/evaluation_htw_data_{}_{}".format(preprocess_typ, corpusname,
                                                      preprocess_typ, model_file_number)
    X_test_unprocessed = pickle.load(
        open("{}X_data_unprocessed.pickle".format(pickle_path), "rb"))

    if os.path.exists("{}testing_data_nn_{}.pickle".format(pickle_path, preprocess_typ)):
        testing_data = pickle.load(
            open("{}testing_data_nn_{}.pickle".format(pickle_path, preprocess_typ), "rb"))
    else:
        print("decode data to word vectors")
        X_test = pickle.load(
            open("{}X_clean_data_{}.pickle".format(pickle_path, preprocess_typ), "rb"))
        y_test = pickle.load(
            open("{}y_data.pickle".format(pickle_path), "rb"))

        sys.path.insert(
            0, "{}/datastories_semeval2017_task4".format(os.getcwd()))
        from utilities.data_loader import Task4Loader
        from utilities.data_loader import get_embeddings

        embeddings, word_indices = get_embeddings(
            corpus="embedtweets.de", dim=200)
        loader = Task4Loader(word_indices, text_lengths=50, loading_data=False,
                             datafolder=pickle_path, preprocess_typ=preprocess_typ)
        testing_data = loader.decode_data_to_embeddings(
            X_test, y_test)  # decode data to word vectors

        pickle.dump(testing_data, open("{}testing_data_nn_{}.pickle".format(
            pickle_path, preprocess_typ), "wb"))

    performance_analysis(testing_data, nn_model, file_name=file_name, file_information=file_information, verbose=True, accuracy=True,
                         confusion_matrix=True, classification_report=True, save_pred=True, X_test_unprocessed=X_test_unprocessed)
