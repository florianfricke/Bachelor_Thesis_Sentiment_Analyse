"""
Created by Florian Fricke.
"""

import sys
sys.path.insert(0, "C:/Users/Flo/Projekte/Bachelor_Thesis_Sentiment_Analyse")
import csv
from utilities.utilities import get_filenames_from_directory
from tqdm import tqdm

def preprocess_scare_corpus(csvfile, tsvfile):
    with open(csvfile, 'r', encoding="utf-8") as csvin, open(tsvfile, 'a', encoding="utf-8", newline='') as tsvout:
        tsvout = csv.writer(tsvout, delimiter='\t')
        data = csvin.readlines()
        data = [line.split("\t") for line in data]
        for line in data:
            rateting = int(line[1])
            line_new = ["",""]

            if rateting < 2:
                line[1] = "negative"
            elif rateting > 4:
                line[1] = "positive"
            else:
                line[1] = "neutral"

            line_new[0] = line[1]
            line_new[1] = line[2] + " " + line[3]
            tsvout.writerow(line_new)
    return

directory = "data/labeled_sentiment_data/scare_app_reviews/"
sentiment_files = get_filenames_from_directory(directory)

for file in tqdm(sentiment_files):
    if(".csv" in file):
        preprocess_scare_corpus(
            directory + file, "data/labeled_sentiment_data/scare_app_reviews.tsv")


