import math
import random
from os.path import isfile
import tensorflow as tf
import pandas as pd
import os
import numpy as np
class DataSet(tf.keras.utils.Sequence):

    def __init__(self, input, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.get_vocab("data/vocab/source_vocab.txt", "data/vocab/target_vocab.txt")
        self.load_data(input)

    def __len__(self):
        return math.ceil(len(self.token_input)/self.batch_size)


    def __getitem__(self, idx):
        random.shuffle(self.token_input)
        x = [source[0] for source in self.token_input]
        y = [target[1] for target in self.token_input]

        batch_x = x[idx * self.batch_size:(idx + 1) *
            self.batch_size]
        batch_y = y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        # print(np.array(batch_x).shape)
        # print(np.array(batch_y)[:, :-1].shape)
        # print(np.array(batch_y)[:, 1:].shape)


        b = [np.array(batch_x), np.array(batch_y)[:, :-1]], np.array(batch_y)[:, 1:]

        return b

    def load_data(self, input):
        # input should contain source and target [[source, target], [], []...]
        source_list = [self.input_tokenize(source[0]) for source in input]
        target_list = [self.target_tokenize(target[1]) for target in input]
        input_list = []
        for source, target in zip(source_list, target_list):
            input_list.append([source, target])
        self.token_input = input_list

    def get_vocab(self, source_vocab_path, target_vocab_path):
        self.source_id2char = {}
        self.source_char2id = {}
        idx = 1
        with open(source_vocab_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                char = line.strip()
                self.source_id2char[idx] = char
                self.source_char2id[char] = idx
                idx += 1
        self.source_id2char[idx] = '[UNK]'
        self.source_char2id['[UNK]'] = idx
        idx += 1
        self.source_id2char[idx] = '[CLS]'
        self.source_char2id['[CLS]'] = idx
        idx += 1
        self.source_id2char[idx] = '[SEP]'
        self.source_char2id['[SEP]'] = idx

        self.target_id2char = {}
        self.target_char2id = {}
        idx = 1
        with open(target_vocab_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                char = line.strip()
                self.target_id2char[idx] = char
                self.target_char2id[char] = idx
                idx += 1
        self.target_id2char[idx] = '[UNK]'
        self.target_char2id['[UNK]'] = idx
        idx += 1
        self.target_id2char[idx] = '[CLS]'
        self.target_char2id['[CLS]'] = idx
        idx += 1
        self.target_id2char[idx] = '[SEP]'
        self.target_char2id['[SEP]'] = idx

    def input_tokenize(self, input):
        token_list = []
        for char in input.split(' '):
            if char in self.source_char2id:
                token_list.append(self.source_char2id[char])
            else:
                token_list.append(self.source_char2id['[UNK]'])
        return self.source_padding(token_list)

    def target_tokenize(self, target):
        token_list = []
        for char in target:
            if char in self.target_char2id:
                token_list.append(self.target_char2id[char])
            else:
                token_list.append(self.target_char2id['[UNK]'])
        return self.target_padding(token_list)
    
    def source_padding(self, input, max_length=8, padding_value=0):
        length = max_length 
        if len(input) >= length:
            input = input[:length]
            input = input
        else:
            padding_length = length - len(input)
            input = input + [padding_value] * padding_length
        return input
    
    def target_padding(self, target, max_length=8, padding_value=0):
        length = max_length -1
        if len(target) >= length:
            target = target[:length]
            target = [self.target_char2id['[CLS]']] + target + [self.target_char2id['[SEP]']]
        else:
            padding_length = length - len(target)
            target = [self.target_char2id['[CLS]']] + target + [padding_value] * padding_length + [self.target_char2id['[SEP]']]
        return target

class Preprocess():
    def __init__(self, file_path):
        self.file_path = file_path
        self.read_file(file_path)
    
    def read_file(self, file):
        df = pd.read_excel(file, engine="openpyxl")
        source_list, target_list = [], []
        for i, row in df.iterrows():
            source = row["source"]
            target = row["target"]
            source_list.append(source)
            target_list.append(target)
        self.source = source_list
        self.target = target_list

    def get_vocab(self, source_vocab_path, target_vocab_path):
        '''
        create source and target vocab
        '''
        if not os.path.isfile(source_vocab_path):
            s = set()
            for source_text in self.source:
                for text in source_text.split(' '):
                    s.add(text)
            with open(source_vocab_path, 'w', encoding='utf-8') as fw:
                for word in s:
                    fw.write(word+'\n')
        if not os.path.isfile(target_vocab_path):
            s = set()
            for target_text in self.target:
                for text in target_text:
                    s.add(text)
            with open(target_vocab_path, 'w', encoding='utf-8') as fw:
                for word in s:
                    fw.write(word+'\n')

class bert_dataset(object):
    def __init__(self, file_name, label_file):
        self.file_name = file_name
        self.label_file = label_file

    def load_file(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as fr:
            for line in fr:
                spline = line.strip().split('\t')
                try:
                    input_text, label = spline
                except Exception as e:
                    print("error splitting data %s" %e)

    def tokenize(self, input_text):
        token_list = []
        for word in input_text:
            if word in self.vocab_dict:
                token_list.append(self.vocab_dict[word])
            else:
                token_list.append('[UNK]')
        return token_list
    def text_to_id(self, input_text):
        pass

    @classmethod
    def data_generator(self):
        input_token = [101, 1, 2, 3, 4, 5, 6, 102]
        input_type_token = [0, 0, 0, 0, 0, 0]
        label_class = 1
        batch_size = 8
        while True:
            x = []
            x1 = []
            x2 = []
            y = []
            for _ in range(100):
                x1.append(input_token)
                x2.append(np.array(input_type_token))
                y.append(np.array(label_class))
                if len(x1) == batch_size:
                    x.append(np.array(x1))
                    yield x, y
                    x = []
                    x1 = []
                    x2 = []
                    y = []






if __name__ == "__main__":
    p = Preprocess("../data/train_data/translate_data.xlsx")
    p.get_vocab("../data/vocab/source_vocab.txt", "../data/vocab/target_vocab.txt")
    # input = [[source, target] for source, target in zip(p.source, p.target)]
    # data = DataSet(input, 2)
    # a=data.__getitem__(0)
    # print(a[0])
