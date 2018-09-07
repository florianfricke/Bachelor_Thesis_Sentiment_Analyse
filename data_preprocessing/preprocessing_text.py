"""
Created by Florian Fricke.
"""

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from textblob_de.lemmatizers import PatternParserLemmatizer
from tqdm import tqdm
from nltk.corpus import stopwords
from utilities.utilities import transform_data
import pickle


class PreprocessingText:
    def __init__(self, text, **kwargs):
        self.text = text
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

    def remove_stopwords(self, data):
        stop_ger = stopwords.words('german')

        allowed_stopwords = ['kein', 'keine', 'keinem',
                             'keinen', 'keiner', 'keines', 'nicht', 'nichts']
        for a in allowed_stopwords:
            stop_ger.remove(a)
        
        customstopwords = ['rt', 'mal', 'heute', 'gerade', 'erst', 'macht', 'eigentlich', 'warum',
                           'gibt', 'gar', 'immer', 'schon', 'beim', 'ganz', 'dass', 'wer', 'mehr', 'gleich', 'wohl']
        normalizedwords = ['<url>', '<email>', '<percent>', 'money>',
                           '<phone>', '<user>', '<time>', '<url>', '<date>', '<number>']
        stop_ger = stop_ger + customstopwords + normalizedwords
        clean_data = []
        if(type(data) == list):
            for d in data:
                data_stop_words = []
                for word in d:
                    if word not in stop_ger:
                        data_stop_words.append(word)
                clean_data.append(data_stop_words)
        if(type(data) == str):
            words = data.split()
            for word in words:
                if word not in stop_ger:
                    clean_data.append(word)
        return clean_data

    def lemmatize_words(self, data):
        _lemmatizer = PatternParserLemmatizer()
        lemmatized_data = []
        if(type(data) == list):
            for d in data:
                text = ""
                for word in d:
                    text = text + " " + word
                l = _lemmatizer.lemmatize(text)
                lemmatized_data.append([i[0] for i in l])
        if(type(data) == str):
            l = _lemmatizer.lemmatize(data)
            lemmatized_data.append([i[0] for i in l])
        return lemmatized_data
    
    def ekphrasis_preprocessing(self):
        X_clean = []
        if(type(self.text) == str):
            X_clean.append(self.text_processor.pre_process_doc(self.text))
        if(type(self.text) == list):
            for row in tqdm(self.text):
                X_clean.append(self.text_processor.pre_process_doc(row))
        return X_clean
