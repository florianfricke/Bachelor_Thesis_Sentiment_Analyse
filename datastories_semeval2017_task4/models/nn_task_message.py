import pickle
import numpy
import glob
import os
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from kutilities.callbacks import MetricsCallback, PlottingCallback
from kutilities.helpers.data_preparation import get_labels_to_categories_map, \
    get_class_weights2, onehot_to_categories
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score
from keras.callbacks import TensorBoard

import sys
sys.path.insert(
    0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse/datastories_semeval2017_task4")

from models.nn_models import build_attention_RNN
from utilities.data_loader import get_embeddings, Task4Loader, prepare_dataset
from evaluate.evaluate import performance_analysis

numpy.random.seed(1337)  # for reproducibility

# specify the word vectors file to use.
WV_CORPUS = "embedtweets.de"
WV_DIM = 200

# Flag that sets the training mode.
# - if FINAL == False,  then the dataset will be split in {train, val, test}
# - if FINAL == True,   then the dataset will be split in {train, val}.
# of the labeled data will be kept for as a validation set for early stopping
FINAL = False

max_length = 50     # max tweet length
DATAFOLDER = "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse/data/labeled_sentiment_data/pickle_files/"
PREPROCESS_TYP = "ekphrasis"

############################################################################
# PERSISTENCE
############################################################################
# if True save model checkpoints, as well as the corresponding word indices
# set PERSIST = True, in order to be able to use the trained model later
PERSIST = True
RESULT_PATH = "results_artificial_neural_network/"

MODEL_FILE_NUMBER = len(
    glob.glob(os.path.join(RESULT_PATH, "model_history_{}*.pickle".format(PREPROCESS_TYP)))) + 1

def best_model(): return "{}{}/model_{}_{}.hdf5".format(
    RESULT_PATH, PREPROCESS_TYP, PREPROCESS_TYP, MODEL_FILE_NUMBER)

def best_model_word_indices(): return "{}model_word_indices_{}.{}.pickle".format(DATAFOLDER, WV_CORPUS, WV_DIM)

############################################################################
# LOAD DATA
############################################################################
embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

if PERSIST:
    if not os.path.exists(best_model_word_indices()):
        pickle.dump(word_indices, open(best_model_word_indices(), 'wb'))

loader = Task4Loader(word_indices, text_lengths=max_length, loading_data=True,
                     datafolder=DATAFOLDER, preprocess_typ=PREPROCESS_TYP)

if FINAL:
    print("\n > running in FINAL mode!\n")
    training, testing = loader.load_final() #Processing Data
else:
    training, validation, testing = loader.load_train_val_test()
    pickle.dump(validation, open("{}validation_data_nn_{}.pickle".format(DATAFOLDER, PREPROCESS_TYP), "wb"))
    # training[0], training[1] = text, sentiment

pickle.dump(training, open("{}training_data_nn_{}.pickle".format(DATAFOLDER, PREPROCESS_TYP), "wb"))
pickle.dump(testing, open("{}testing_data_nn_{}.pickle".format(DATAFOLDER, PREPROCESS_TYP), "wb"))

############################################################################
# NN MODEL
############################################################################
print("Building NN Model...")
attention_model = "simple" # simple, None
nn_model = build_attention_RNN(embeddings, classes=3, max_length=max_length,    #classes = pos., neg, neutral
                               unit=LSTM, layers=2, cells=150,
                               bidirectional=True,
                               attention=attention_model,  
                               noise=0.3,
                               final_layer=False,
                               dropout_final=0.5,
                               dropout_attention=0.5, #0.5
                               dropout_words=0.3,
                               dropout_rnn=0.3,
                               dropout_rnn_U=0.3,
                               clipnorm=1, lr=0.001, loss_l2=0.0001)   # gradient clipping and learning rate                           
print(nn_model.summary())

############################################################################
# CALLBACKS
############################################################################
metrics = {
    "f1_pn": (lambda y_test, y_pred:
              f1_score(y_test, y_pred, average='macro',
                       labels=[class_to_cat_mapping['positive'],
                               class_to_cat_mapping['negative']])),
    "M_recall": (
        lambda y_test, y_pred: recall_score(y_test, y_pred, average='macro')),
    "M_precision": (
        lambda y_test, y_pred: precision_score(y_test, y_pred,
                                               average='macro'))
}

classes = ['positive', 'negative', 'neutral']
class_to_cat_mapping = get_labels_to_categories_map(
    classes)  # {'negative': 0, 'neutral': 1, 'positive': 2}
cat_to_class_mapping = {v: k for k, v in
                        get_labels_to_categories_map(classes).items()}  # {0: 'negative', 1: 'neutral', 2: 'positive'}

_datasets = {}
_datasets["1-train"] = training,
_datasets["2-val"] = validation if not FINAL else testing
if not FINAL:
    _datasets["3-test"] = testing

metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
plotting = PlottingCallback(grid_ranges=(0.5, 0.75), height=5,
                            benchmarks={"SE17": 0.681}, plot_name="model_{}_{}".format(PREPROCESS_TYP, MODEL_FILE_NUMBER))
tensorboard = TensorBoard(log_dir='./logs')

_callbacks = []
_callbacks.append(metrics_callback)
_callbacks.append(tensorboard)
_callbacks.append(plotting)

if PERSIST:
    monitor = "val_loss" # 'val.macro_recall' 
    mode = "min" # mode="max"
    checkpointer = ModelCheckpoint(filepath=best_model(),
                                   monitor=monitor, mode=mode,  
                                   verbose=1, save_best_only=True)
    _callbacks.append(checkpointer)

############################################################################
# APPLY CLASS WEIGHTS
############################################################################
class_weights = get_class_weights2(onehot_to_categories(training[1]),
                                   smooth_factor=0)
print("Class weights:",
      {cat_to_class_mapping[c]: w for c, w in class_weights.items()})

# 50-50
epochs = 50
batch_size = 10

history = nn_model.fit(training[0], training[1],
                       validation_data=validation if not FINAL else testing,
                       epochs=epochs, batch_size=batch_size,  
                       class_weight=class_weights, callbacks=_callbacks)

pickle.dump(history.history, open("{}{}/model_history_{}_{}.pickle".format(
    RESULT_PATH, PREPROCESS_TYP, PREPROCESS_TYP, MODEL_FILE_NUMBER), "wb"))

############################################################################
# Evaluation
############################################################################
file_name = "{}/evaluation_{}_{}".format(PREPROCESS_TYP, PREPROCESS_TYP, MODEL_FILE_NUMBER)
file_information = "epochs = " + str(epochs) + "\nbatch_size = " + str(batch_size) + \
    "\nmax textlength = " + str(max_length) + "\npreprocess-typ = " + \
    PREPROCESS_TYP + "\nattention model = " + \
    str(attention_model) + "\nbest model with " + mode + " " + monitor
file_information = file_information + "\ndropout_attention = 0.5\n3 LSTM Layer"
performance_analysis(testing, nn_model, file_name=file_name, file_information=file_information, verbose=True, accuracy=True,
                     confusion_matrix=True, classification_report=True)
