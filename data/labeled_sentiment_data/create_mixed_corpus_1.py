from utilities.utilities import save_data
from utilities.utilities import get_random_elements_from_list
from random import shuffle
import pickle

path = "data/labeled_sentiment_data/pickle_files/"
corpus_name1 = "sb10k_and_one_million_posts_corpus"
corpus_name2 = "scare"
new_corpus_name = "mixed_corpus_1"

dataset1 = pickle.load(
    open("{}{}/dataset_unprocessed.pickle".format(path, corpus_name1), "rb"))

dataset2 = pickle.load(
    open("{}{}/dataset_unprocessed.pickle".format(path, corpus_name2), "rb"))

pos_scare = [d for d in dataset2 if d[0]=="positive"]
neg_scare = [d for d in dataset2 if d[0]=="negative"]

pos_scare_random = get_random_elements_from_list(pos_scare, 8330)
neg_scare_random = get_random_elements_from_list(neg_scare, 7325)

new_dataset = dataset1 + pos_scare_random + neg_scare_random
shuffle(new_dataset)

print("positive words: {}".format(len([d for d in new_dataset if d[0] == "positive"])))
print("negative words: {}".format(len([d for d in new_dataset if d[0] == "negative"])))
print("neutral words: {}".format(len([d for d in new_dataset if d[0] == "neutral"])))

X = [d[1] for d in new_dataset]  # text
y = [d[0] for d in new_dataset]  # sentiment

save_data(new_dataset, path=path + new_corpus_name + "/",
            filename="dataset_unprocessed")
save_data(X, path=path + new_corpus_name + "/",
            filename="X_data_unprocessed")
save_data(y, path=path + new_corpus_name + "/",
          filename="y_data")


