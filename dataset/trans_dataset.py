# coding=utf-8
'''
Generate data to train the model
'''
from random import shuffle
import numpy as np
from config import CONFIG
import os

class dataset(object):
    def __init__(self,data_type=None):
        self.data_type=data_type
        self.load_vocab()


    def read_from_file(self, file_path):
        corpus_list = []
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                spline = line.split('\t')
                training_corpus = [char for char in spline[0]]
                training_corpus = ['[CLS]'] + training_corpus + ['[SEP]']
                multilabels = []
                try:
                    for label in spline[1:]:
                        if label != 'NULL':
                            if '-' not in label:
                                # if no '-', it is a domain label
                                multilabels.append(self.domain2id[label])
                            else:
                                multilabels.append(self.intent2id[label])

                    corpus_list.append([training_corpus, multilabels])
                except:
                    print("abort this data as there is an error")

        self.corpus_list = corpus_list


    def data_generator(self):
        while True:
            shuffle(self.corpus_list)
            x = []
            input_id_lst = []
            input_type_lst = []
            domain_list = []
            intent_list = []
            for data_pair in self.corpus_list:
                corpus, labels = data_pair
                input_id_lst.append(self.pad_to_max_seq(self.sent_to_idx(corpus)))
                input_type_lst.append(self.get_input_type(corpus))
                domain_one_hot_array = np.zeros(CONFIG.DomainLabelSize, dtype='int32')
                intent_one_hot_array = np.zeros(CONFIG.IntentLabelSize, dtype='int32')
                for domain_labels, intent_labels in labels:
                    for label_idx in domain_labels:
                        domain_one_hot_array[label_idx] = 1
                    for label_idx in intent_labels:
                        intent_one_hot_array[label_idx] = 1
                    domain_list.append(domain_one_hot_array)
                    intent_list.append(intent_one_hot_array)
                if len(input_id_lst) == CONFIG.BatchSize and len(domain_list) == CONFIG.BatchSize and len(intent_list) == CONFIG.BatchSize:
                    x.append(np.array(input_id_lst))
                    x.append(np.array(input_type_lst))
                    yield x, [domain_list, intent_list]
                    x = []
                    domain_list = []
                    intent_list = []
                    input_id_lst = []
                    input_type_lst = []

    def get_input_type(self, text):
        input_type_lst = []
        sep_index = text.index('[SEP]')
        first_part = [0]*sep_index
        input_type_lst.append(self.pad_to_max_seq(first_part, padding=0, max_seq=100))
        return input_type_lst[0]



    def sent_to_idx(self, sent):
        vectorized = []
        if isinstance(sent, str):
            for char in sent:
                vectorized.append(self.char2id[char] if char in self.char2id else self.char2id['[UNK]'])
        elif isinstance(sent, list):
            for char in sent:
                vectorized.append(self.char2id[char] if char in self.char2id else self.char2id['[UNK]'])
        return vectorized

    def load_vocab(self):
        self.id2char = dict()
        self.char2id = dict()
        idx = 0
        with open(CONFIG.VocabFilePath, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                self.char2id[line] = idx
                self.id2char[idx] = line
                idx += 1

    def load_labels(self, domain_path, intent_path):
        self.domain2id = {}
        self.id2domain = {}
        self.intent2id = {}
        self.id2intent = {}
        domain_idx = 0
        intent_idx = 0
        with open(domain_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                self.domain2id[line] = domain_idx
                self.id2domain[domain_idx] = line
                domain_idx += 1
        with open(intent_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                self.intent2id[line] = intent_idx
                self.id2intent[intent_idx] = line
                intent_idx += 1


    def pad_to_max_seq(self, lst, padding=0, max_seq=100):
        if len(lst) > max_seq:
            return lst[:max_seq]
        else:
            paddings = max_seq - len(lst)
            return lst + [padding] * paddings

    def to_predict(self, inputs):
        chars = ['[CLS]'] + [char for char in inputs] + ['[SEP]']
        sent = self.pad_to_max_seq(self.sent_to_idx(chars))
        input_type_list = [0] * 100

        return [np.array([sent]), np.array([input_type_list])]

    def inverse_transform(self, pred, threshold=None, topk=None):
        results = []
        scores = []
        for i in range(len(pred)):
            int_labels = np.argsort(pred[i])
            for j in range(len(int_labels)-1, -1, -1):
                idx = int_labels[j]
                if threshold and float(pred[i][idx]) >= threshold:
                    results.append(self.id2domain[idx])
                    scores.append(pred[i][idx])
                if topk:
                    if len(results) == topk:
                        return results, scores
                    else:
                        results.append(self.id2domain[idx])
                        scores.append(pred[i][idx])
        return results, scores


from Tokenization import FullTokenizer

class DATASET_MULTILINGUAL(dataset):
    def __init__(self, vocab_dir, data_type=None):
        self.data_type = data_type
        self.tokenizer = FullTokenizer(vocab_file=vocab_dir, do_lower_case=False)
        self.load_vocab()
        super(DATASET_MULTILINGUAL, self).__init__()

    def read_from_file(self, file_path):
        corpus_list = []
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                spline = line.split('\t')
                subject, context = spline[0].split('###')[0], ''.join(spline[0].split('###')[1:])
                subject, context = self.tokenizer.tokenize(subject), self.tokenizer.tokenize(context)
                training_corpus = ['[CLS]'] + subject + ['[SEP]'] + context+['[SEP]']
                multilabels = []
                try:
                    for label in spline[1:]:
                        if label != 'NULL':
                            multilabels.append(self.domain2id[label])
                    corpus_list.append([training_corpus, multilabels])
                except:
                    print(line)
                    print("abort this data as there is an error")
                    input()

        self.corpus_list = corpus_list

    def to_predict(self, inputs):
        subject, context = inputs.split('###')
        subject, context = self.tokenizer.tokenize(subject), self.tokenizer.tokenize(context)
        chars = ['[CLS]'] + subject + ['[SEP]'] + context + ['[SEP]']
        sent = self.pad_to_max_seq(self.sent_to_idx(chars))
        input_type_lst=self.get_input_type(chars)

        return [np.array([sent]), np.array([input_type_lst])]

class DataSet_MonoData(dataset):
    def __init__(self, vocab_dir, data_type=None):
        self.data_type = data_type
        self.tokenizer = FullTokenizer(vocab_file=vocab_dir, do_lower_case=False)
        self.load_vocab()
        super(DataSet_MonoData, self).__init__()
    
    def read_from_multi_files(self, dir_path_list):
        corpus_list = []
        data_count = 0
        for file in dir_path_list:
            print("current file read : %s" %file)
            with open(file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    spline = line.split('\t')
                    uq = spline[0]
                    uq_tokenized = self.tokenizer.tokenize(uq)
                    uq_tokenized = self.concat_custom_token_representation(uq_tokenized)
                    training_corpus = ['[CLS]'] + uq_tokenized +['[SEP]']
                    multilabels = []
                    data_count += 1
                    labels = spline[1:]
                    if "other-other" in labels:
                        a = True
                    try:
                        domain_labels = []
                        intent_labels = []
                        for label in spline[1:]:
                            domain = label.split('-')[0]
                            domain_labels.append(self.domain2id[domain])
                            intent_labels.append(self.intent2id[label])
                        if len(domain_labels) == 0 or len(intent_labels) == 0:
                            print(line)
                            print("error does not labels")
                            continue
                        multilabels.append([domain_labels, intent_labels])
                        corpus_list.append([training_corpus, multilabels])
                    except Exception as e:
                        print(line.split('\t'))
                        print("abort this data as there is an error %s"%e)

        self.corpus_list = corpus_list
        self.count = data_count



    def read_from_file(self, file_path):
        corpus_list = []
        data_count = 0
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                spline = line.split('\t')
                uq = spline[0]
                uq_tokenized = self.tokenizer.tokenize(uq)
                # uq_tokenized = self.concat_custom_token_representation(uq_tokenized)
                training_corpus = ['[CLS]'] + uq_tokenized +['[SEP]']
                multilabels = []
                data_count += 1
                labels = spline[1:]
                try:
                    domain_labels = []
                    intent_labels = []
                    for label in labels:
                        domain = label.split('-')[0]
                        domain_labels.append(self.domain2id[domain])
                        intent_labels.append(self.intent2id[label])
                    if len(domain_labels) == 0 or len(intent_labels) == 0:
                        print(line)
                        print("error does not labels")
                        continue
                    multilabels.append([domain_labels, intent_labels])
                    corpus_list.append([training_corpus, multilabels])
                except Exception as e:
                    print(line.split('\t'))
                    print("abort this data as there is an error %s"%e)

        self.corpus_list = corpus_list
        self.count = data_count

    def concat_custom_token_representation(self, tokenized_input):
        target_token_representation = ["[city]", "[province]", "[poi]", "[time]", "[country]", "[district]"]
        potential_token = ""
        token_idx = 0
        new_tokenized_input = []
        while token_idx < len(tokenized_input):
            token = tokenized_input[token_idx]
            if token == '[':
                tmp_token = []
                while token_idx < len(tokenized_input) and tokenized_input[token_idx] != ']':
                    token = tokenized_input[token_idx]
                    potential_token += token
                    tmp_token.append(token)
                    token_idx += 1
                if token_idx < len(tokenized_input):
                    if tokenized_input[token_idx] == ']':
                        potential_token += ']'
                    tmp_token.append(tokenized_input[token_idx])
                if potential_token in target_token_representation:
                    token = potential_token
                    new_tokenized_input.append(token)
                else:
                    new_tokenized_input.extend(tmp_token)
            else:
                new_tokenized_input.append(token)
            token_idx += 1
        return new_tokenized_input




    def to_predict(self, inputs):
        #subject, context = inputs.split('###')
        uq = inputs
        uq_tokenized = self.tokenizer.tokenize(uq)
        print(111111111)
        print(uq_tokenized)
        chars = ['[CLS]'] + uq_tokenized + ['[SEP]']
        sent = self.pad_to_max_seq(self.sent_to_idx(chars))
        input_type_lst=self.get_input_type(chars)

        return [np.array([sent]), np.array([input_type_lst])]

    def inverse_transform(self, pred, topk=1):
        domain_result = []
        intent_result = []
        # domain_pred, intent_pred = pred
        domain_pred = []
        intent_pred = pred
        domain_pred = np.squeeze(domain_pred)
        intent_pred = np.squeeze(intent_pred)

        domain_pred_sorted = np.argsort(domain_pred)
        for i in range(len(domain_pred_sorted)-1, -1, -1):
            idx = domain_pred_sorted[i]
            domain_name = self.id2domain[idx]
            domain_score = domain_pred[idx]
            domain_result.append([domain_name, domain_score])
            if topk and len(domain_result) == topk:
                break

        intent_pred_sorted = np.argsort(intent_pred)
        for j in range(len(intent_pred_sorted)-1,-1, -1):
            idx = intent_pred_sorted[j]
            # if idx >= 23:
            #    continue
            intent_name = self.id2intent[idx]
            intent_score = intent_pred[idx]
            intent_result.append([intent_name, intent_score])
            if topk and len(intent_result) == topk:
                break
        return domain_result, intent_result

class DataSet_Single_label(dataset):
    def __init__(self, vocab_dir, data_type=None):
        self.data_type = data_type
        self.tokenizer = FullTokenizer(vocab_file=vocab_dir)
        self.load_vocab()
        super(DataSet_Single_label, self).__init__()
    
    def read_from_multi_files(self, dir_path):
        corpus_list = []
        data_count = 0
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            print("current file read : %s" %file_path)
            with open(file_path, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = line.strip()
                    spline = line.split('\t')
                    uq = spline[0]
                    uq_tokenized = self.tokenizer.tokenize(uq)
                    uq_tokenized = self.concat_custom_token_representation(uq_tokenized)
                    training_corpus = ['[CLS]'] + uq_tokenized +['[SEP]']
                    multilabels = []
                    data_count += 1
                    labels = spline[1:]
                    if "other-other" in labels:
                        a = True
                    try:
                        domain_labels = []
                        intent_labels = []
                        for label in spline[1:]:
                            domain = label.split('-')[0]
                            domain_labels.append(self.domain2id[domain])
                            intent_labels.append(self.intent2id[label])
                        if len(domain_labels) == 0 or len(intent_labels) == 0:
                            print(line)
                            print("error does not labels")
                            continue
                        multilabels.append([domain_labels, intent_labels])
                        corpus_list.append([training_corpus, multilabels])
                    except Exception as e:
                        print(line.split('\t'))
                        print("abort this data as there is an error %s"%e)

        self.corpus_list = corpus_list
        self.count = data_count



    def read_from_file(self, file_path):
        corpus_list = []
        data_count = 0
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                spline = line.split('\t')
                uq = spline[0]
                uq_tokenized = self.tokenizer.tokenize(uq)
                training_corpus = ['[CLS]'] + uq_tokenized +['[SEP]']
                multilabels = []
                data_count += 1
                labels = spline[1:]
                if "other-other" in labels:
                    a = True
                try:
                    labels = []
                    for label in spline[1:]:
                        domain = label.split('-')[0]
                        labels.append(self.intent2id[label])
                    if len(labels) == 0:
                        print(line)
                        print("error does not labels")
                        continue
                    multilabels.append(labels)
                    corpus_list.append([training_corpus, multilabels])
                except Exception as e:
                    print(line.split('\t'))
                    print("abort this data as there is an error %s"%e)

        self.corpus_list = corpus_list
        self.count = data_count

    def concat_custom_token_representation(self, tokenized_input):
        target_token_representation = ["[city]", "[province]", "[poi]", "[time]", "[country]", "[district]"]
        potential_token = ""
        token_idx = 0
        new_tokenized_input = []
        while token_idx < len(tokenized_input):
            token = tokenized_input[token_idx]
            if token == '[':
                tmp_token = []
                while token_idx < len(tokenized_input) and tokenized_input[token_idx] != ']':
                    token = tokenized_input[token_idx]
                    potential_token += token
                    tmp_token.append(token)
                    token_idx += 1
                if token_idx < len(tokenized_input):
                    if tokenized_input[token_idx] == ']':
                        potential_token += ']'
                    tmp_token.append(tokenized_input[token_idx])
                if potential_token in target_token_representation:
                    token = potential_token
                    new_tokenized_input.append(token)
                else:
                    new_tokenized_input.extend(tmp_token)
            else:
                new_tokenized_input.append(token)
            token_idx += 1
        return new_tokenized_input




    def to_predict(self, inputs):
        #subject, context = inputs.split('###')
        uq = inputs
        uq_tokenized = self.tokenizer.tokenize(uq)
        chars = ['[CLS]'] + uq_tokenized + ['[SEP]']
        sent = self.pad_to_max_seq(self.sent_to_idx(chars))
        input_type_lst=self.get_input_type(chars)

        return [np.array([sent]), np.array([input_type_lst])]

    def inverse_transform(self, pred, topk=1):
        domain_result = []
        intent_result = []
        domain_pred, intent_pred = pred
        domain_pred = np.squeeze(domain_pred)
        intent_pred = np.squeeze(intent_pred)

        domain_pred_sorted = np.argsort(domain_pred)
        for i in range(len(domain_pred_sorted)-1, -1, -1):
            idx = domain_pred_sorted[i]
            domain_name = self.id2domain[idx]
            domain_score = domain_pred[idx]
            if domain_score >= 0.5:
                domain_result.append([domain_name, domain_score])
            if topk and len(domain_result) == topk:
                break

        intent_pred_sorted = np.argsort(intent_pred)
        for j in range(len(intent_pred_sorted)-1,-1, -1):
            idx = intent_pred_sorted[j]
            # if idx >= 23:
            #    continue
            intent_name = self.id2intent[idx]
            intent_score = intent_pred[idx]
            if intent_score >= 0.4:
                intent_result.append([intent_name, intent_score])
            if topk and len(intent_result) == topk:
                break
        return domain_result, intent_result

    def load_labels(self, label_path):
        self.id2char = {}
        self.char2id = {}
        idx = 0
        with open(label_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                self.id2char[line] = idx
                self.char2id[idx] = line
                idx += 1

class DataSet_trilabels():
    def __init__(self, vocab_file, data_type, model=None, do_lower_case=True, label_path=[]):
        self.tokenizer=FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.model = model
        self.data_type=data_type
        self.load_labels(label_path)
        self.load_vocab(vocab_file)

    def load_labels(self, label_path):
        if not isinstance(label_path, list):
            raise TypeError ("label path has to be a list. And it could contain multiple label files")
        label_clusters = list()
        for i, label_file in enumerate(label_path):
            label2id = dict()
            id2label = dict()
            with open(label_file, 'r', encoding='utf-8') as fr:
                idx = 0
                for line in fr:
                    line = line.strip()
                    label2id[line] = idx
                    id2label[idx] = line
                    idx += 1
            label_clusters.append({"label2id":label2id, "id2label":id2label})
        self.label_clusters = label_clusters
        print(self.label_clusters)


    def read_from_file(self, file_path):
        corpus_list = []
        corpus_int_list = []
        cnt = 0
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                spline = line.split('\t')
                try:
                    data, label1, label2, label3 = spline[0], spline[1], spline[2], spline[3:]
                    corpus_list.append([data, [label1, label2, label3]])
                    corpus_int_list.append([self.sent_to_idx(data), [self.label_clusters[0]["label2id"][label1],
                                                                      self.label_clusters[1]["label2id"][label2],
                                                                      [self.label_clusters[2]["label2id"][label] for label in label3]]])
                    cnt += 1
                    # if cnt > 65:
                    #     break
                except Exception as e:
                    print("abort this data as there is an error %s" %e)

        self.corpus_list = corpus_list
        self.corpus_int_list = corpus_int_list
        self.count = len(corpus_int_list)

    def eval_and_update(self):
        inputs = []
        input_types = []
        true_labels = []
        acc_log = []
        shuffle(self.corpus_int_list)
        for corpus_int in self.corpus_int_list:
            int_data, labels_idx = corpus_int
            if len(inputs)== CONFIG.BatchSize:
                batch_pred = self.model.predict([inputs, input_types])
                overall_acc = self.batch_acc(true_labels, batch_pred)
                acc_log.append(overall_acc)
                inputs, input_types, true_labels = [], [], []
            else:
                inputs.append(int_data)
                input_types.append([0]*len(int_data))
                true_labels.append(labels_idx)
        average_total_acc = sum(acc_log)/len(acc_log)
        return average_total_acc

    def batch_acc(self, y_true, y_pred):
        scene_pred_list, intent_pred_list, semantic_pred_list = [], [], []
        scene_true_list, intent_true_list, semantic_true_list = [], [], []
        scene, intent, semantic = y_pred
        scene_pred_list, intent_pred_list, semantic_pred_list = np.argsort(scene), np.argsort(intent), np.argsort(semantic)
        for true_labels in y_true:
            intent_one_hot_array = np.zeros(CONFIG.IntentLabelSize, dtype='int32')
            domain_one_hot_array = np.zeros(CONFIG.DomainLabelSize, dtype='int32')
            semantic_one_hot_array = np.zeros(CONFIG.SemanticLabelSize, dtype='int32')
            domain_label, intent_label, semantic_label = true_labels
            domain_one_hot_array[domain_label] = 1
            intent_one_hot_array[intent_label] = 1
            for semantic in semantic_label:
                semantic_one_hot_array[semantic] = 1
            scene_true_list.append(domain_one_hot_array)
            intent_true_list.append(intent_one_hot_array)
            semantic_true_list.append(semantic_one_hot_array)

        acc = 0
        for i in range(len(scene_pred_list)):
            scene_pred = scene_pred_list[i][-1]
            scene_true = np.argsort(scene_true_list[i])[-1]
            if scene_pred == scene_true:
                intent_pred = intent_pred_list[i][-1]
                intent_true = np.argsort(intent_true_list[i])[-1]
                if intent_pred == intent_true:
                    semantic_pred = semantic_pred_list[i][-1]
                    semantic_true = np.argsort(semantic_true_list[i])[-1]
                    if semantic_pred == semantic_true:
                        acc += 1
        return acc/CONFIG.BatchSize

    def data_generator(self):
        while True:
            shuffle(self.corpus_int_list)
            x = []
            input_id_lst = []
            input_type_lst = []
            domain_list = []
            intent_list = []
            semantic_list = []
            for data_pair in self.corpus_int_list:
                int_data, labels_idx = data_pair
                input_id_lst.append(int_data)
                input_type_lst.append([0]*len(int_data))
                domain_one_hot_array = np.zeros(CONFIG.DomainLabelSize, dtype='int32')
                intent_one_hot_array = np.zeros(CONFIG.IntentLabelSize, dtype='int32')
                semantic_one_hot_array = np.zeros(CONFIG.SemanticLabelSize, dtype='int32')
                domain_label, intent_label, semantic_label = labels_idx
                domain_one_hot_array[domain_label] = 1
                intent_one_hot_array[intent_label] = 1
                for semantic in semantic_label:
                    semantic_one_hot_array[semantic] = 1
                domain_list.append(domain_one_hot_array)
                intent_list.append(intent_one_hot_array)
                semantic_list.append(semantic_one_hot_array)
                if len(input_id_lst) == CONFIG.BatchSize and len(intent_list) == CONFIG.BatchSize:
                    x.append(np.array(input_id_lst))
                    x.append(np.array(input_type_lst))
                    yield x, [intent_list]
                    x = []
                    domain_list = []
                    intent_list = []
                    semantic_list = []
                    input_id_lst = []
                    input_type_lst = []


    def sent_to_idx(self, input_text):
        tokens = self.tokenizer.tokenize(input_text)
        tokens = self.pad_to_max_seq(tokens)
        token_list = []
        for token in tokens:
            if token == 0:
                token_list.append(0)
            else:
                token_list.append(self.char2id[token] if token in self.char2id else self.char2id['[UNK]'])
        return token_list

    def pad_to_max_seq(self, tokens, padding_value=0, padding_length=100):
        max_length = padding_length - 2
        if len(tokens) <= max_length:
            return ['[CLS]'] + tokens + ['[SEP]'] + [0] * (max_length - len(tokens))
        else:
            return ['[CLS]'] + tokens[:max_length] + ['[SEP]']

    def load_vocab(self, vocab_file):
        self.id2char = dict()
        self.char2id = dict()
        idx = 1
        with open(vocab_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                self.id2char[idx] = line
                self.char2id[line] = idx
                idx += 1

    def to_predict(self, input_text):
        tokens =  self.sent_to_idx(input_text)
        return [[tokens], [[0]*100]]

    def inverse_transform(self, pred, topk=1):
        # scene, intent, semantic = pred
        # scene, intent, semantic = np.squeeze(scene), np.squeeze(intent), np.squeeze(semantic)
        # scene_arg_idx = np.argsort(scene)
        intent, semantic = pred
        intent_arg_idx = np.argsort(np.squeeze(intent))
        semantic_arg_idx = np.argsort(np.squeeze(semantic))
        scene_arg_idx= []

        scene_pred = None
        for i in range(len(scene_arg_idx)-1, -1, -1):
            scene_pred = self.label_clusters[0]["id2label"][scene_arg_idx[i]]
            break


        intent_pred = None
        for i in range(len(intent_arg_idx)-1, -1, -1):
            intent_pred = self.label_clusters[1]["id2label"][intent_arg_idx[i]]
            break
        

        semantic_pred = None
        for i in range(len(semantic_arg_idx)-1, -1, -1):
            semantic_pred = self.label_clusters[2]["id2label"][semantic_arg_idx[i]]
            break
        return intent_pred, semantic_pred
            




if __name__ == "__main__":
    test = DataSet_MonoData(vocab_dir='model/chinese_L-12_H-768_A-12/vocab.txt', data_type='train')
    test.load_labels('data/domain_and_intent/domain_cat.txt', 'data/domain_and_intent/intent_cat.txt')
    test.read_from_multi_files("data/domain_and_intent/raw_data/20230718/train2/train_files")
    test.data_generator()
    a = test.sent_to_idx(['[CLS]']+[char for char in "这是一个测试"]+['[SEP]'])
    print(a)











