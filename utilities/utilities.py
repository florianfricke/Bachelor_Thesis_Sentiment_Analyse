from os import listdir
from os.path import isfile, join
import html

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
