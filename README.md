# Bachelor Thesis - Evaluierung von Machine-Learning-Systemen zur Sentiment-Analyse

## Voraussetzungen
`pip install -r requirements.txt` <br>
`python - m textblob.download_corpora` <br>

Der Pfad muss noch in jeder Datei angepasst werden, damit Python mit den erstellten Packages arbeiten kann. (wird noch von mir geändert)
`sys.path.insert(0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")`

## Daten
Die Word-Embeddings können unter folgendem Link heruntergeladen werden:
https://www.spinningbytes.com/resources/wordembeddings/ <br>
Embeddings in `datastories_semeval2017_task4/embeddings` kopieren und den Namen in folgendes Format bringen `{}.{}d` z.B. `embedtweets.de.200d`

Die folgenden Daten befinden sich un- und vorverarbeitet im Repository und stammen von folgenden Quellen:
GermanPolarityClues - A Lexical Resource for German Sentiment Analysis <br>
http://www.ulliwaltinger.de/sentiment/

One Million Posts Corpus <br>
https://ofai.github.io/million-post-corpus/

SB-10k: German Sentiment Corpus <br>
https://www.spinningbytes.com/resources/germansentiment/

Daten-Vorverarbeitung:
Es wird das Text-Processing-Tool ekphrasis verwendet.
https://github.com/cbaziotis/ekphrasis

## Sentiment Analyse mit Künstlichem Neuronalem Netz
Die Implementierung von Data Stories Semeval 2017 Sentiment Analysis mit Twitterdaten und Neuronalen Netzen befindet sich im Verzeichnis `datastories_semeval2017_task4`. <br>
Diese Implementierungsvariante wurde auf deutsche Textdaten von mir angepasst.<br>
https://github.com/cbaziotis/datastories-semeval2017-task4

## Evaluierung
folgt..<br>
Durch Ausführung des Konsolenbefehls `tensorboard --logdir logs` im Projektordner können verschiedene Metriken während und nach dem Training evaluiert werden.
