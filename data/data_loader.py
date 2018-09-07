"""
Created by Florian Fricke.
"""

import os
from utilities.utilities import clean_text
from os.path import join

class DataLoader:
    def __init__(self):
        self._dirname = os.path.dirname(__file__)

    def get_data(self, filename, seperator="\t", sentiment_column_number=0, text_column_number=1):
        data = list()

        for line_id, line in enumerate(
                open(join(self._dirname, filename), "r", encoding="utf-8").readlines()):
            try:
                columns = line.rstrip().split(seperator)
                sentiment = columns[sentiment_column_number]
                text = clean_text(" ".join(columns[text_column_number:]))
                if text != "Not Available":
                    data.append((sentiment, text))

            except Exception:
                print("\nWrong format in line:{} in file:{}".format(
                    line_id, filename))
                raise Exception
        return data

