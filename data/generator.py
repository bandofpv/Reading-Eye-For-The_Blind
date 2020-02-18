"""
Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time.
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences

import data.preproc as pp
import h5py
import numpy as np
import unicodedata


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, source, batch_size, charset, max_text_length, predict=False):
        self.tokenizer = Tokenizer(charset, max_text_length)
        self.batch_size = batch_size
        self.partitions = ['test'] if predict else ['train', 'valid', 'test']

        self.size = dict()
        self.steps = dict()
        self.index = dict()
        self.dataset = dict()

        with h5py.File(source, "r") as f:
            for pt in self.partitions:
                self.dataset[pt] = dict()
                self.dataset[pt]['dt'] = f[pt]['dt'][:]
                self.dataset[pt]['gt'] = f[pt]['gt'][:]

        for pt in self.partitions:
            # decode sentences from byte
            self.dataset[pt]['gt'] = [x.decode() for x in self.dataset[pt]['gt']]

            # set size and setps
            self.size[pt] = len(self.dataset[pt]['gt'])
            self.steps[pt] = int(np.ceil(self.size[pt] / self.batch_size))
            self.index[pt] = 0

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.index['train'] >= self.size['train']:
                self.index['train'] = 0

            index = self.index['train']
            until = self.index['train'] + self.batch_size
            self.index['train'] = until

            x_train = self.dataset['train']['dt'][index:until]
            y_train = self.dataset['train']['gt'][index:until]

            x_train = pp.augmentation(x_train,
                                      rotation_range=1.5,
                                      scale_range=0.05,
                                      height_shift_range=0.025,
                                      width_shift_range=0.05,
                                      erode_range=5,
                                      dilate_range=3)

            x_train = pp.normalization(x_train)

            y_train = [self.tokenizer.encode(y) for y in y_train]
            y_train = pad_sequences(y_train, maxlen=self.tokenizer.maxlen, padding="post")

            yield (x_train, y_train, [])

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        while True:
            if self.index['valid'] >= self.size['valid']:
                self.index['valid'] = 0

            index = self.index['valid']
            until = self.index['valid'] + self.batch_size
            self.index['valid'] = until

            x_valid = self.dataset['valid']['dt'][index:until]
            y_valid = self.dataset['valid']['gt'][index:until]

            x_valid = pp.normalization(x_valid)

            y_valid = [self.tokenizer.encode(y) for y in y_valid]
            y_valid = pad_sequences(y_valid, maxlen=self.tokenizer.maxlen, padding="post")

            yield (x_valid, y_valid, [])

    def next_test_batch(self):
        """Return model predict parameters"""

        while True:
            if self.index['test'] >= self.size['test']:
                self.index['test'] = 0
                break

            index = self.index['test']
            until = self.index['test'] + self.batch_size
            self.index['test'] = until

            x_test = self.dataset['test']['dt'][index:until]
            x_test = pp.normalization(x_test)

            yield x_test


class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = pp.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "")
