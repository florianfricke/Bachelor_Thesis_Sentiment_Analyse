import sys
import os
sys.path.insert(
    0, "{}/datastories_semeval2017_task4".format(os.getcwd()))
from evaluate.evaluate import performance_analysis
from utilities_nn.data_loader import Task4Loader
from utilities_nn.data_loader import get_embeddings
from keras.models import load_model
from kutilities.layers import Attention
import pickle

############################################################################
# Evaluate Data
############################################################################
evaluate_corpus_name = "htw"
pickle_path = "data/labeled_sentiment_data/pickle_files/{}/".format(
    evaluate_corpus_name)
preprocess_typ = "ekphrasis"
model_file_number = 3
file_information = ""
attention_mechanism = True
corpusname = "mixed_corpus_1"

print("load model_{}_{}".format(preprocess_typ, model_file_number))
if(attention_mechanism):
    nn_model = load_model(
        'results_artificial_neural_network/{}/{}/model_{}_{}.hdf5'.format(
            preprocess_typ, corpusname, preprocess_typ, model_file_number),
        custom_objects={'Attention': Attention})
else:
    nn_model = load_model(
        'results_artificial_neural_network/{}/{}/model_{}_{}.hdf5'.format(
            preprocess_typ, corpusname, preprocess_typ, model_file_number))

file_name = "{}/{}/evaluation_{}_{}_{}".format(preprocess_typ, corpusname, evaluate_corpus_name,
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
