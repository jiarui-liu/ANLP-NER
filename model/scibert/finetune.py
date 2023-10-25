import os
import torch
from transformers import AutoTokenizer, BertForTokenClassification, AutoConfig
from torch.utils.data import DataLoader
import nltk
import re

# for model training labels
tag2id = {
    'O': 0,
    'MethodName': 1,
    'HyperparameterName': 2,
    'HyperparameterValue': 3,
    'MetricName': 4,
    'MetricValue': 5,
    'TaskName': 6,
    'DatasetName': 7,
}

id2tag = {v: k for k, v in tag2id.items()}

# for postprocessing
tag2span_tag = {
    'MethodName': ['B-MethodName', 'I-MethodName'],
    'HyperparameterName': ['B-HyperparameterName', 'I-HyperparameterName'],
    'HyperparameterValue': ['B-HyperparameterValue', 'I-HyperparameterValue'],
    'MetricName': ['B-MetricName', 'I-MetricName'],
    'MetricValue': ['B-MetricValue', 'I-MetricValue'],
    'TaskName': ['B-TaskName', 'I-TaskName'],
    'DatasetName': ['B-DatasetName', 'I-DatasetName'],
}

def set_seed(seed):
    import random
    import os

    if seed is None:
        from efficiency.log import show_time
        seed = int(show_time())

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def reevaluate_sentence_boundary(lines, processed):
    doc = None
    if processed:
        doc = " ".join([re.split("\s+", line)[0] for line in lines if len(re.split("\s+", line)) > 0])
    else:
        doc = " ".join([re.split("\s+", line)[0] for line in lines if len(re.split("\s+", line)) > 0 and '-DOCSTART- -X-' not in line])
    sents = nltk.sent_tokenize(doc)
    return sents

def process_conll_file(file_path):
    """Process the original conll file
    """
    # format the conll file
    flag = False
    processed = True
    lines = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.strip() != "":
                lines.append(line)
            else:
                flag = True
            if "-DOCSTART- -X-" in line:
                processed = False
    if flag:
        with open(file_path, 'w') as f:
            f.write("".join(lines))

    # format the data structure
    # sentences in a file
    words_dir, tags_dir = [], []
    # words in a sentences
    new_words, new_tags = [], []
    if not processed:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            sents = reevaluate_sentence_boundary(lines, processed)
            idx = 0
            for line in lines:
                line = re.split('\s+', line.strip())
                if len(line) == 4:
                    if " ".join(new_words + [line[0]]) == sents[idx]:
                        # new sentence
                        new_words.append(line[0])
                        new_tags.append(line[-1])
                        words_dir.append(new_words)
                        tags_dir.append(new_tags)
                        new_words, new_tags = [], []
                        idx += 1
                    else:
                        # this sentence
                        new_words.append(line[0])
                        new_tags.append(line[-1])
        
    else:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            sents = reevaluate_sentence_boundary(lines, processed)
            idx = 0
            for line in lines:
                line = re.split('\s+', line.strip())
                if len(line) == 2:
                    if " ".join(new_words + [line[0]]) == sents[idx]:
                        # new sentence
                        new_words.append(line[0])
                        new_tags.append(line[-1])
                        words_dir.append(new_words)
                        tags_dir.append(new_tags)
                        new_words, new_tags = [], []
                        idx += 1
                    else:
                        # this sentence
                        new_words.append(line[0])
                        new_tags.append(line[-1])

    # print(len(words_dir))
    return words_dir, tags_dir

def build_df(obj):
    import pandas as pd
    df = pd.DataFrame()
    for k in obj:
        df[k] = obj[k]
    return df

class TuneSciBERT():
    def __init__(self, train_dir, val_dir, test_dir, seed=0):
        # set seed
        set_seed(seed=seed)
        # dataset
        self.train = self.prepare_data(train_dir)
        self.val = self.prepare_data(val_dir)
        self.test = self.prepare_data(test_dir)
        # device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        
        
    def load_pretrained_model(self, num_classes, original=False, path=None):
        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
        config.num_labels = num_classes
        config.label2id = tag2id
        config.id2label = id2tag
        print(config)
        
        if original:
            self.model = BertForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
        else:
            # "/home/jiaruil5/anlp/model/save/lr_5e-05_bs_16_epoch_0.pth"
            assert path is not None
            # self.model.load_state_dict(torch.load(path))
            self.model = BertForTokenClassification.from_pretrained(path, config=config)
        
        self.model.to(self.device)
        
    def prepare_data(self, dir):
        res_dict = {
            'words': [],
            'tags': [],
        }
        for filename in os.listdir(dir):
            words, tags = process_conll_file(os.path.join(dir, filename))
            res_dict['words'].extend(words)
            res_dict['tags'].extend(tags)
        return res_dict
    
    def build_dataset(self, split='train'):
        pass
        # self.train_data = 
        # self.train_loader = 