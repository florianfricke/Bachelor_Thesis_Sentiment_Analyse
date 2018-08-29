"""
Created by Florian Fricke.
"""

from os import listdir
from os.path import isfile, join
import html
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

def get_filenames_from_directory(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files

def clean_text(text):
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text

def transform_data(data):
    transformed_data = []
    for d in data:
        sentence = ""
        for index, word in enumerate(d):
            if(len(d)-1 == index):
                sentence = sentence + word
            else:
                sentence = sentence + word + " "
        transformed_data.append(sentence)
    return transformed_data #return format: ['sentence','sentence']

def save_data(data, path="", filename=""):
    pickle.dump(data, open("{}{}.pickle".format(path, filename), "wb"))
    return None

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

    plt.tight_layout()
    plt.ylabel('Tatsächliche Klasse')
    plt.xlabel('Vorhergesagte Klasse')
