# Bachelor Thesis - Evaluierung von Machine-Learning-Systemen zur Sentiment-Analyse

## Voraussetzungen
`pip install -r requirements.txt` <br>
`python - m textblob.download_corpora` <br>

## Daten
Die Word-Embeddings können unter folgendem Link heruntergeladen werden:<br>
https://www.spinningbytes.com/resources/wordembeddings/ <br>
Danach die Embeddings in `datastories_semeval2017_task4/embeddings` kopieren und den Namen in folgendes Format bringen `{}.{}d` z.B. `embedtweets.de.200d`

Die folgenden Daten befinden sich unvorverarbeitet und vorverarbeitet im Repository und stammen von folgenden Quellen: <br>
GermanPolarityClues - A Lexical Resource for German Sentiment Analysis <br>
http://www.ulliwaltinger.de/sentiment/

One Million Posts Corpus <br>
https://ofai.github.io/million-post-corpus/

SB-10k: German Sentiment Corpus <br>
https://www.spinningbytes.com/resources/germansentiment/
<br>

Es wurde zu einem gewissen Teil ein weiterer Textdatensatz verwendet. (scare) <br>
http://www.romanklinger.de/scare/ <br>
Dieser Datensatz ist nicht öffentlich verfügbar, kann jedoch beim Entwickler auf Anfrage erhalten werden. Die Daten konnten aufgrund der beschränkten Bearbeitungszeit nicht mehr evaluiert werden.
<br>

Daten-Vorverarbeitung:<br>
Es wird das Textvorverarbeitungstool ekphrasis verwendet.
https://github.com/cbaziotis/ekphrasis
<br><br>
Die Daten unterscheiden sich in ihren Vorverarbeitungsschritten. <br>
Varianten der Textdaten: ekphrasis, ekphrasis und Stoppwörterentfernung, ekphrasis und Stoppwörterentfernung und Lemmatisierung<br>


## Sentiment Analyse mit Künstlichem Neuronalem Netz
Die angepasste Implementierung von Data Stories für den Semeval 2017 mit Twitterdaten und Neuronalen Netzen befindet sich im Verzeichnis `datastories_semeval2017_task4`. <br>
Diese Implementierungsvariante wurde von mir auf deutsche Textdaten angepasst.<br>
Die Datei `datastories_semeval2017_task4/models/nn_task_message.py` startet das Training und die Evaluierung des Neuronalen Netzes<br>
https://github.com/cbaziotis/datastories-semeval2017-task4

## weitere Sentiment Analyse Methoden
`preprocess_evaluate_corpus.py` startet das Training und die Evaluierung des Wörterbuchbasierten Sentiment-Analyse-Systems, das Sentiment-Analyse-System der Python-Bibliothek textblob-de und den Multinomial-Naive-Bayes-Klassifikator.
Weiterhin können durch Aufruf dieser Python-Datei die Daten vorverarbeitet werden.

## Evaluierung
Durch Ausführung des Konsolenbefehls `tensorboard --logdir logs` im Projektordner können verschiedene Metriken während und nach dem Training evaluiert werden.<br>
Die Daten wurden je nach Vorverarbeitungsschritten und verschiedenen Parametern evaluiert.