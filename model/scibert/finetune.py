import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, Trainer
from torch.utils.data import DataLoader, Dataset
import nltk
import re
import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

full_tag2tag = {
    'O': 'O',
    'B-MethodName': 'MethodName',
    'I-MethodName': 'MethodName',
    'B-HyperparameterName': 'HyperparameterName',
    'I-HyperparameterName': 'HyperparameterName',
    'B-HyperparameterValue': 'HyperparameterValue',
    'I-HyperparameterValue': 'HyperparameterValue',
    'B-MetricName': 'MetricName',
    'I-MetricName': 'MetricName',
    'B-MetricValue': 'MetricValue',
    'I-MetricValue': 'MetricValue',
    'B-TaskName': 'TaskName',
    'I-TaskName': 'TaskName',
    'B-DatasetName': 'DatasetName',
    'I-DatasetName': 'DatasetName',
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

def compute_metrics(eval_prediction):
    # https://medium.com/@rakeshrajpurohit/customized-evaluation-metrics-with-hugging-face-trainer-3ff00d936f99
    preds, labels = eval_prediction
    preds = preds.argmax(-1)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average='weighted'),
        "recall": recall_score(labels, preds, average='weighted'),
        "f1": f1_score(labels, preds, average='weighted'),
    }
    

class TuneDataset(Dataset):
    def __init__(self, data, tokenizer, device='cpu'):
        self.words = data['words']
        self.tags = data['tags']
        self.device=device
        
        self.words_tokenized = tokenizer(
            self.words,
            truncation=True,
            padding=True,
            is_split_into_words=True,
        )
        
        # construct a token level label list
        labels = []
        for idx, token in enumerate(self.words_tokenized.input_ids):
            word_level_label = self.tags[idx]
            # print(word_level_label)
            label = []
            for word_id in self.words_tokenized.word_ids(idx):
                if word_id is None:
                    label.append(-100)
                else:
                    label.append(tag2id[full_tag2tag[word_level_label[word_id]]])
            labels.append(label)
                        
                        
        self.x = self.words_tokenized.input_ids
        self.y = labels

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        # print(x)
        # print(y, type(y))
        return {
            "input_ids": torch.Tensor(x).to(torch.long),
            "labels": torch.Tensor(y).to(torch.long),
        }


class TuneSciBERT():
    def __init__(self, train_dir, val_dir, test_dir, seed=0):
        # set seed
        set_seed(seed=seed)
        # dataset
        self._train = self.prepare_data(train_dir)
        self._val = self.prepare_data(val_dir)
        self._test = self.prepare_data(test_dir)
        # device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'allenai/scibert_scivocab_uncased',
            model_max_length=512,
        )
        
        
    def load_pretrained_model(self, num_classes, original=False, path=None):
        config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
        config.num_labels = num_classes
        config.label2id = tag2id
        config.id2label = id2tag
        print(config)
        
        if original:
            self.model = AutoModelForTokenClassification.from_pretrained('allenai/scibert_scivocab_uncased', config=config)
        else:
            # "/home/jiaruil5/anlp/model/save/lr_5e-05_bs_16_epoch_0.pth"
            assert path is not None
            # self.model.load_state_dict(torch.load(path))
            self.model = AutoModelForTokenClassification.from_pretrained(path, config=config)
        
        # self.model.to(self.device)
        
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
    
    def build_dataset(self):
        self.train_data = TuneDataset(self._train, self.tokenizer, self.device)
        self.val_data = TuneDataset(self._val, self.tokenizer, self.device)
        self.test_data = TuneDataset(self._test, self.tokenizer, self.device)
        
    def train(self, training_args):
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
        )
        self.trainer.train()
    
    def eval(self):
        results = self.trainer.evaluate()
        print(results)