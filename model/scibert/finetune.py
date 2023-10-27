import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoConfig,
    Trainer,
)
from torch.utils.data import DataLoader, Dataset
import nltk
import re
import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import evaluate


# for model training labels
tag2id = {
    "O": 0,
    "MethodName": 1,
    "HyperparameterName": 2,
    "HyperparameterValue": 3,
    "MetricName": 4,
    "MetricValue": 5,
    "TaskName": 6,
    "DatasetName": 7,
}

id2tag = {v: k for k, v in tag2id.items()}

# for postprocessing
tag2span_tag = {
    "MethodName": ["B-MethodName", "I-MethodName"],
    "HyperparameterName": ["B-HyperparameterName", "I-HyperparameterName"],
    "HyperparameterValue": ["B-HyperparameterValue", "I-HyperparameterValue"],
    "MetricName": ["B-MetricName", "I-MetricName"],
    "MetricValue": ["B-MetricValue", "I-MetricValue"],
    "TaskName": ["B-TaskName", "I-TaskName"],
    "DatasetName": ["B-DatasetName", "I-DatasetName"],
}

full_tag2tag = {
    "O": "O",
    "B-MethodName": "MethodName",
    "I-MethodName": "MethodName",
    "B-HyperparameterName": "HyperparameterName",
    "I-HyperparameterName": "HyperparameterName",
    "B-HyperparameterValue": "HyperparameterValue",
    "I-HyperparameterValue": "HyperparameterValue",
    "B-MetricName": "MetricName",
    "I-MetricName": "MetricName",
    "B-MetricValue": "MetricValue",
    "I-MetricValue": "MetricValue",
    "B-TaskName": "TaskName",
    "I-TaskName": "TaskName",
    "B-DatasetName": "DatasetName",
    "I-DatasetName": "DatasetName",
}

metric = evaluate.load('seqeval')

def set_seed(seed):
    import random
    import os

    if seed is None:
        from efficiency.log import show_time

        seed = int(show_time())

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    doc = " ".join(
        [re.split("\s+", line)[0] for line in lines if len(re.split("\s+", line)) > 0]
    )
    sents = nltk.sent_tokenize(doc)
    return sents


def process_conll_file(file_path):
    """Process the original conll file"""
    # format the conll file
    flag = False
    processed = True
    lines = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            if line.strip() != "":
                lines.append(line)
            else:
                flag = True
            if "-DOCSTART- -X-" in line:
                processed = False
    if flag:
        with open(file_path, "w") as f:
            f.write("".join(lines))

    # format the data structure
    # sentences in a file
    words_dir, tags_dir = [], []
    # words in a sentences
    new_words, new_tags = [], []
    if not processed:
        with open(file_path, "r") as f:
            lines = f.readlines()
            sents = reevaluate_sentence_boundary(lines, processed)
            idx = 0
            for line in lines:
                line = re.split("\s+", line.strip())
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
        with open(file_path, "r") as f:
            lines = f.readlines()
            sents = reevaluate_sentence_boundary(lines, processed)
            idx = 0
            for line in lines:
                line = re.split("\s+", line.strip())
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

def select_major_vote(pred_tmp):
    pred_sel = None
    counts = {}
    for l in pred_tmp:
        counts[l] = counts.get(l, 0) + 1
    counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
    for idx, k in enumerate(counts):
        if idx == 0:
            if k != 0:
                pred_sel = k
                break
            else:
                continue
        if idx == 1:
            pred_sel = k
    if pred_sel is None:
        pred_sel = 0
    return pred_sel

def compute_metrics(eval_prediction):
    # https://medium.com/@rakeshrajpurohit/customized-evaluation-metrics-with-hugging-face-trainer-3ff00d936f99
    _preds, _labels = eval_prediction
    _preds = _preds.argmax(-1)
    # print(preds)
    # print(labels)
    preds, labels = [], []
    for pred, label in zip(_preds, _labels):
        for p, l in zip(pred, label):
            if l != -100:
                preds.append(p)
                labels.append(l)
    # metrics = metric.compute(predictions=preds, references=labels)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "f1": f1_score(labels, preds, average="weighted"),
    }
    # return {
    #     "accuracy": metrics['overall_accuracy'],
    #     "precision": metrics['overall_precision'],
    #     "recall": metrics['overall_recall'],
    #     "f1": metrics['overall_f1'],
    # }


def csv_2_conll(csv_file_path):
    csv_df = pd.read_csv(csv_file_path)

    coNLL_format_data = []
    for index, row in csv_df.iterrows():
        word = row["input"]
        coNLL_format_data.append(f"{word}\tO")

    coNLL_file_path = csv_file_path.replace("csv", "conll")
    with open(coNLL_file_path, "w") as coNLL_file:
        coNLL_file.write("\n".join(coNLL_format_data))


def conll_2_csv(conll_file_path):
    csv_file_path = conll_file_path.replace(".conll", ".csv")
    with open(conll_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    headers = ["id", "input", "target"]
    rows = []
    for line in lines:
        line = line.strip()
        if line:
            fields = line.split()
            row = {headers[i]: field for i, field in enumerate(fields)}
            rows.append(row)

    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TuneDataset(Dataset):
    def __init__(self, data, tokenizer, device="cpu"):
        self.words = data["words"]
        self.tags = data["tags"]
        self.device = device

        self.words_tokenized = tokenizer(
            self.words,
            truncation=True,
            padding=True,
            is_split_into_words=True,
        )

        # construct a token level label list
        labels = []
        self.word_ids = []
        for idx, token in enumerate(self.words_tokenized.input_ids):
            word_level_label = self.tags[idx]
            # print(word_level_label)
            label = []
            _word_id = []
            for word_id in self.words_tokenized.word_ids(idx):
                if word_id is None:
                    label.append(-100)
                    _word_id.append(-100)
                else:
                    label.append(tag2id[full_tag2tag[word_level_label[word_id]]])
                    _word_id.append(word_id)
            labels.append(label)
            self.word_ids.append(_word_id)

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
            "word_ids": torch.Tensor(self.word_ids[index]).to(torch.long),
        }


class TuneSciBERT:
    def __init__(self, train_dir, val_dir, test_dir, seed=0):
        # set seed
        set_seed(seed=seed)
        # dataset
        self._train = self.prepare_data(train_dir)
        self._val = self.prepare_data(val_dir)
        # self._test = self.prepare_data(test_dir)
        # device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased",
            model_max_length=512,
        )

    def load_pretrained_model(self, num_classes, original=False, path=None):
        config = AutoConfig.from_pretrained("allenai/scibert_scivocab_uncased")
        config.num_labels = num_classes
        config.label2id = tag2id
        config.id2label = id2tag
        print(config)

        if original:
            self.model = AutoModelForTokenClassification.from_pretrained(
                "allenai/scibert_scivocab_uncased", config=config
            )
        else:
            # "/home/jiaruil5/anlp/model/save/lr_5e-05_bs_16_epoch_0.pth"
            assert path is not None
            # self.model.load_state_dict(torch.load(path))
            self.model = AutoModelForTokenClassification.from_pretrained(
                path, config=config
            )

        # self.model.to(self.device)

    def prepare_data(self, dir):
        res_dict = {
            "words": [],
            "tags": [],
        }
        for filename in os.listdir(dir):
            words, tags = process_conll_file(os.path.join(dir, filename))
            res_dict["words"].extend(words)
            res_dict["tags"].extend(tags)
        return res_dict

    def prepare_test_data(self, file):
        df = pd.read_csv(file)
        _dict = df.to_dict(orient='records')
        res_dict = {
            'words': [],
            'tags': [],
        }
        
        res_words, res_tags = [], []
        for idx, l in enumerate(_dict):
            if type(l['input']) == str:
                res_words.append(l['input'])
                res_tags.append('O')
                if idx == len(_dict) - 1:
                    res_dict['words'].append(res_words)
                    res_dict['tags'].append(res_tags)
            else:
                res_dict['words'].append(res_words)
                res_dict['tags'].append(res_tags)
                res_words, res_tags = [], []
        return res_dict, df

    def build_dataset(self):
        self.train_data = TuneDataset(self._train, self.tokenizer, self.device)
        self.val_data = TuneDataset(self._val, self.tokenizer, self.device)
        # self.test_data = TuneDataset(self._test, self.tokenizer, self.device)

    def train(self, training_args):
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
        )
        self.batch_size = training_args.per_device_train_batch_size
        self.trainer.train()

    def eval(self):
        results = self.trainer.evaluate()
        print(results)
    
    def generate_prediction(self, test_data = None):
        self.model.eval()
        if test_data is None:
            test_data = TuneDataset(self._test, self.tokenizer, self.device)
        
        final_preds = []
        # not batched
        for i, item in tqdm(enumerate(test_data)):
            
            with torch.no_grad():
                outputs = self.model(item['input_ids'].reshape(1, -1).to(self.device))
            
            words = test_data.words[i]
            final_pred = []
            
            pred_tmp = []
            word_id_tmp = 0
            # print(f"word_ids, {item['word_ids']}, input_ids, {len(item['input_ids'])}")
            for label, input_id, word_id, logit in zip(item['labels'], item['input_ids'], item['word_ids'], outputs.logits[0]):
                pred = torch.argmax(logit)
                pred = pred.item()
                label = label.item()
                input_id = input_id.item()
                word_id = word_id.item()
                
                # print(f'pred: {pred}, label: {label}, input_id: {input_id}, word_id: {word_id}')
                
                if word_id != -100:
                    if word_id != word_id_tmp:
                        # get label
                        pred_sel = select_major_vote(pred_tmp)
                        
                        for i in range(word_id-word_id_tmp):
                            final_pred.append(id2tag[pred_sel])
                        
                        pred_tmp = [pred]
                        word_id_tmp = word_id
                    else:
                        pred_tmp.append(pred)
                elif len(pred_tmp) > 0:
                    pred_sel = select_major_vote(pred_tmp)
                    final_pred.append(id2tag[pred_sel])
                    pred_tmp = []
            #     print(f"final_pred: {final_pred}, length: {len(final_pred)}")
            # print(f"words: {words}, final_pred: {final_pred}")
            while (len(final_pred) < len(words)):
                final_pred.append("O")
            assert len(words) == len(final_pred)
            
            final_preds.append(final_pred)
        return final_preds, test_data.tags, test_data.words

    
    def test(self, test_file, test_out):
        self._test, df = self.prepare_test_data(test_file)
        self.test_data = TuneDataset(self._test, self.tokenizer, self.device)
        preds, _, words = self.generate_prediction(test_data=self.test_data)
        preds = self.map_back(preds)
        
        sent_idx = 0
        word_idx = 0
        df = df.astype({'target': 'str'})
        print(df.dtypes)
        for idx, row in df.iterrows():
            if pd.isnull(row['input']):
                sent_idx += 1
                word_idx = 0
                df.at[idx, 'target'] = 'O'
                continue
            df.at[idx, 'target'] = preds[sent_idx][word_idx]
            word_idx += 1
        headers = ['id', 'target']
        df.to_csv(test_out, columns=headers, index=False)
    
    def get_classification_report(self, preds, labels):
        metrics = metric.compute(predictions=self.map_back(preds), references=labels)

        print({
            "accuracy": metrics['overall_accuracy'],
            "precision": metrics['overall_precision'],
            "recall": metrics['overall_recall'],
            "f1": metrics['overall_f1'],
        })
        preds = [tag2id[j] for i in preds for j in i]
        labels = [tag2id[full_tag2tag[j]] for i in labels for j in i]
        print(classification_report(labels, preds))
    
    def map_back(self, preds):
        # preds: a list of list (sentences)
        first_seen=True
        store_label = 'O'
        final_preds = []
        for pred_list in preds:
            final_pred_list = []
            for idx, pred in enumerate(pred_list):
                if pred == 'O':
                    final_pred_list.append(pred)
                    store_label = 'O'
                else:
                    if pred != store_label:
                        first_seen = True
                    else:
                        first_seen = False
                    
                    if first_seen:
                        final_pred_list.append(tag2span_tag[pred][0])
                        store_label = pred
                    else:
                        final_pred_list.append(tag2span_tag[pred][1])
            final_preds.append(final_pred_list)
        return final_preds