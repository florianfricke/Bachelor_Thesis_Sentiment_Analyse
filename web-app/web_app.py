"""
Created by Florian Fricke.
"""
import sys
import os
sys.path.insert(
    0, "{}/web-app".format(os.getcwd()))

from flask import Flask, render_template, request, jsonify
from predictions.predict import predict_sentiment
from keras.models import load_model
from kutilities.layers import Attention
import numpy as np
import pickle

app = Flask(__name__)

def load_nn_model(model_file_number=1, attention_mechanism=True, corpusname="", preprocess_typ="ekphrasis"):
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
    return nn_model

nn_model1 = load_nn_model(model_file_number=1, attention_mechanism=True,
                       corpusname="sb10k_and_one_million_posts_corpus", preprocess_typ="ekphrasis")
nn_model2 = load_nn_model(model_file_number=3, attention_mechanism=True,
                       corpusname="mixed_corpus_1", preprocess_typ="ekphrasis")

word_indices = pickle.load(
    open("data/labeled_sentiment_data/pickle_files/model_word_indices_embedtweets.de.200.pickle", "rb"))

text = "Ich liebe Sie"
data = [text]
result = predict_sentiment(
    data, "ekphrasis", nn_model1, nn_model2, word_indices)
print(result)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predicton", methods=['GET','POST'])
def prediction():
    text = request.args.get('text', 0, type=str)
    data = [text]
    result = predict_sentiment(data, "ekphrasis", nn_model1, nn_model2, word_indices)
    for i in range(len(result)):
        if result[i] == 0:
            result[i] = "negative"
        if result[i] == 1:
            result[i] = "neutral"
        if result[i] == 2:
            result[i] = "positive"
    return jsonify(text=text, result=result)

if __name__ == '__main__':
    app.run(debug=True, port=8085)



