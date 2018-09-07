"""
Created by Florian Fricke.
"""

import csv

def preprocess_one_million_posts_corpus(csvfile, tsvfile):
    with open(csvfile, 'r', encoding="utf-8") as csvin, open(tsvfile, 'w', encoding="utf-8", newline='') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t',)
        
        for line in csvin:
            if line[0] == "SentimentNeutral":
                line[0] = "neutral"
            elif line[0] == "SentimentNegative":
                line[0] = "negative"
            elif line[0] == "SentimentPositive":
                line[0] = "positive"
            else:
                continue

            line[1] = line[1].replace('\r', '').replace('\n', '')
            tsvout.writerow(line)
    return

preprocess_one_million_posts_corpus(
    "data/labeled_sentiment_data/million_pos_corpus.csv", "data/labeled_sentiment_data/million_pos_corpus.tsv")

