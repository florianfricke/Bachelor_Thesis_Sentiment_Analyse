"""
Created by Florian Fricke.
"""

import os
from utilities.utilities import clean_text
from os.path import join

def remove_identical_lines(file_name, target_file_name, seperator="\t", sentiment_column_number=0, text_column_number=1):
    data = list()
    dirname = os.path.dirname(__file__)
    with open(join(dirname, target_file_name), 'w', encoding="utf-8") as f:
        for line_id, line in enumerate(
                open(join(dirname, file_name), "r", encoding="utf-8").readlines()):
            try:
                columns = line.rstrip().split(seperator)
                sentiment = columns[sentiment_column_number]
                text = clean_text(" ".join(columns[text_column_number:]))
                if text not in data:
                    print(sentiment + seperator + text, file=f)
                    data.append(text)

            except Exception:
                print("\nWrong format in line:{} in file:{}".format(
                    line_id, file_name))
                raise Exception
    return data

remove_identical_lines(
    "join_data.tsv", "join_data_clean.tsv")



