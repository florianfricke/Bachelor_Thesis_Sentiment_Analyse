"""
Created by Christos Baziotis.
"""
import random
import pickle
import numpy

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from kutilities.helpers.data_preparation import print_dataset_statistics, \
    labels_to_categories, categories_to_onehot
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from embeddings.WordVectorsManager import WordVectorsManager
from modules.CustomPreProcessor import CustomPreProcessor
from modules.EmbeddingsExtractor import EmbeddingsExtractor

def prepare_dataset(X, y, pipeline, y_one_hot=True, y_as_is=False):
    X = pipeline.fit_transform(X)
    if y_as_is:
        try:
            return X, numpy.asarray(y, dtype=float)
        except:
            return X, y

    # 1 - Labels (positive) to categories (integer)
    y_cat = labels_to_categories(y)

    if y_one_hot:
        # 2 - Labels to one-hot vectors
        return X, categories_to_onehot(y_cat)

    return X, y_cat

def get_embeddings(corpus, dim):
    vectors = WordVectorsManager(corpus, dim).read()
    vocab_size = len(vectors)
    print('Loaded %s word vectors.' % vocab_size)
    wv_map = {}
    pos = 0
    # +1 for zero padding token and +1 for <unk>
    emb_matrix = numpy.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        pos = i + 1
        wv_map[word] = pos
        emb_matrix[pos] = vector

    pos += 1
    wv_map["<unk>"] = pos
    emb_matrix[pos] = numpy.random.uniform(low=-0.05, high=0.05, size=dim)

    return emb_matrix, wv_map

def prepare_text_only_dataset(X, pipeline):
    X = pipeline.fit_transform(X)
    return X

class Task4Loader:
    def __init__(self, word_indices, text_lengths, loading_data=True, datafolder="", preprocess_typ="ekphrasis", **kwargs):
        self.word_indices = word_indices
        self.y_one_hot = kwargs.get("y_one_hot", True)

        self.pipeline = Pipeline([
            ('ext', EmbeddingsExtractor(word_indices=word_indices,
                                        max_lengths=text_lengths,
                                        add_tokens=(True),
                                        unk_policy="random"))])
        if(loading_data):
            print("Loading data...")
            self.X_train = pickle.load(open(
                "{}X_train_{}.pickle".format(datafolder, preprocess_typ), "rb"))
            self.X_test = pickle.load(open(
                "{}X_test_{}.pickle".format(datafolder, preprocess_typ), "rb"))
            self.y_train = pickle.load(open(
                "{}y_train_{}.pickle".format(datafolder, preprocess_typ), "rb"))
            self.y_test = pickle.load(open(
                "{}y_test_{}.pickle".format(datafolder, preprocess_typ), "rb"))

            print("-------------------\ntraining set stats\n-------------------")
            print_dataset_statistics(self.y_train)
            print("-------------------")

    def load_train_val_test(self):
        X_val, X_test, y_val, y_test = train_test_split(self.X_test, self.y_test,
                                                        test_size=0.5,
                                                        stratify=self.y_test,
                                                        random_state=42)
        print("\nPreparing training set...")
        training = prepare_dataset(self.X_train, self.y_train, self.pipeline,
                                    self.y_one_hot)
        print("\nPreparing validation set...")
        validation = prepare_dataset(X_val, y_val, self.pipeline,
                                        self.y_one_hot)
        print("\nPreparing test set...")
        testing = prepare_dataset(X_test, y_test, self.pipeline,
                                  self.y_one_hot)
        return training, validation, testing

    def load_final(self):                                               
        print("\nPreparing training set...")
        training = prepare_dataset(self.X_train, self.y_train, self.pipeline,
                                   self.y_one_hot)            
        print("\nPreparing test set...")
        testing = prepare_dataset(self.X_test, self.y_test, self.pipeline,
                                  self.y_one_hot)
        return training, testing
    
    def decode_data_to_embeddings(self, X_data, y_data):
        embedding_data = prepare_dataset(X_data, y_data, self.pipeline,
                                    self.y_one_hot)
        return embedding_data
