"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(
    0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from textblob_de.lemmatizers import PatternParserLemmatizer
from tqdm import tqdm
from nltk.corpus import stopwords
from data.data_loader import DataLoader
from utilities.utilities import save_data
from utilities.utilities import transform_data
import pickle

class PreprocessingCorpus:
    def __init__(self, datafolders, save_data_path, **kwargs):
        self.datafolders = datafolders
        self.dataset = self.get_data()
        self.X = [d[1] for d in self.dataset]  # text
        self.y = [d[0] for d in self.dataset]  # sentiment
        self.corpus_name = kwargs.get('corpus_name', "")
    
        save_data(self.dataset, path=save_data_path, filename="{}dataset_unprocessed".format(self.corpus_name))
        save_data(self.X, path=save_data_path,
                  filename="{}X_data_unprocessed".format(self.corpus_name))
        save_data(self.y, path=save_data_path, filename="{}y_data".format(self.corpus_name))

        self.text_processor = TextPreProcessor(
            # terms that will be normalize e.g. test@gmx.de to <email>
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                    'time', 'date', 'number'],

            # terms that will be annotated e.g. <hashtag>#test</hashtag>
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                    'emphasis'},
            fix_html=True,  # fix HTML tokens

            unpack_hashtags=True,  # perform word segmentation on hashtags

            # select a tokenizer. You can use SocialTokenizer, or pass your own if not text tokenized on whitespace
            # the tokenizer, should take as input a string and return a list of tokens 
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )

    def get_data(self):
        dataset = list()
        for file in self.datafolders:
            dataset.extend(DataLoader().get_data(
                file[0], file[1], file[2], file[3]))
        return dataset                                                

    def remove_stopwords(self, data):
        stop_ger = stopwords.words('german')
        customstopwords = ['rt', 'mal', 'heute', 'gerade', 'erst', 'macht', 'eigentlich', 'warum',
                        'gibt', 'gar', 'immer', 'schon', 'beim', 'ganz', 'dass', 'wer', 'mehr', 'gleich', 'wohl']
        normalizedwords = ['<url>', '<email>', '<percent>', 'money>',
                        '<phone>', '<user>', '<time>', '<url>', '<date>', '<number>']
        stop_ger = stop_ger + customstopwords + normalizedwords
        clean_data = []

        for d in data:
            data_stop_words = []
            for word in d:
                if word not in stop_ger:
                    data_stop_words.append(word)
            clean_data.append(data_stop_words)
        return clean_data

    # def remove_empty_data(self, data):
    #     data_clean = []
    #     for d in data:
    #         if d != "":
    #             data_clean.append(d)
    #     return data_clean

    def lemmatize_words(self, data):
            _lemmatizer = PatternParserLemmatizer()
            lemmatized_data = []
            for d in tqdm(data):
                text = ""
                for word in d:
                    text = text + " " + word
                l = _lemmatizer.lemmatize(text)
                lemmatized_data.append([i[0] for i in l])
            return lemmatized_data

    def ekphrasis_preprocessing(self):
        X_clean = []
        for row in tqdm(self.X):
            X_clean.append(self.text_processor.pre_process_doc(row))
        return X_clean

    def save_clean_data(self, clean_data, path, preprocess_typ):
        save_data(clean_data, path=path,
                  filename="{}X_clean_data_{}".format(self.corpus_name, preprocess_typ))
                  

if __name__ == '__main__':
    path = "data/labeled_sentiment_data/pickle_files/"
    preprocess_typ = "lemmatized"
    def lemmatize_words(data):
            _lemmatizer = PatternParserLemmatizer()
            lemmatized_data = []
            for d in tqdm(data):
                text = ""
                for word in d:
                    text = text + " " + word
                l = _lemmatizer.lemmatize(text)
                lemmatized_data.append([i[0] for i in l])
            return lemmatized_data
    data = pickle.load(
        open("{}X_clean_data_stopwords.pickle".format(path), "rb"))
    clean_data = lemmatize_words(data)
    pickle.dump(clean_data, open(
        "{}X_clean_data_lemmatized.pickle".format(path), "wb"))
