import errno
import os
import pickle
import numpy
from utilities_nn.ResourceManager import ResourceManager

class WordVectorsManager(ResourceManager):
    def __init__(self, corpus=None, dim=None, omit_non_english=False):
        super().__init__()
        self.omit_non_english = omit_non_english
        self.wv_filename = "{}.{}d".format(corpus, str(dim))
        self.parsed_filename = "{}.{}d.pickle".format(corpus, str(dim))
    
    def is_ascii(self, text):
        try:
            text.encode('ascii')
            return True
        except:
            return False

    def write(self):
        _word_vector_file = os.path.join(os.path.dirname(__file__), self.wv_filename) # return file directory

        if os.path.exists(_word_vector_file):
            print('Indexing file {} ...'.format(self.wv_filename))
            embeddings_dict = {}

            with open(_word_vector_file, "r", encoding="utf-8") as file:
                for i, line in enumerate(file):
                    if line.strip() != "" or line != "\n": # or len(line) > 0
                        values = line.split()
                        word = values[0]
                        coefs = numpy.asarray(values[1:], dtype='float32')

                        if word.lower() in {'<unk>', "<unknown>"}:
                            print(word)
                            print("UNKNOWN")
                            print()

                        if self.omit_non_english and not self.is_ascii(word):
                            continue
                        if word not in embeddings_dict or word.strip() == "":
                            embeddings_dict[word] = coefs
                            # 'House': array([0.174788, 0.091168, -0.317676,...])
            print('Found %s word vectors.' % len(embeddings_dict))

            # save Embeddings into a pickle-File
            with open(os.path.join(os.path.dirname(__file__), self.parsed_filename), 'wb') as pickle_file:
                pickle.dump(embeddings_dict, pickle_file)

        else:
            print("{} not found!".format(_word_vector_file))
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), _word_vector_file)

    # load pickle file
    def read(self):
        _parsed_file = os.path.join(os.path.dirname(__file__), self.parsed_filename)
        if os.path.exists(_parsed_file):  # pickle file for Embeddings available
            with open(_parsed_file, 'rb') as f:
                return pickle.load(f)
        else:  # no pickle file for Embeddings available
            self.write()
            return self.read()
