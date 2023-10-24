from transformers import AutoTokenizer, BertForPreTraining
import torch
import os
import json
import nltk
from nltk.tokenize import sent_tokenize
import sys
import random
import numpy as np
import pandas as pd
from transformers import AdamW
from tqdm import tqdm

sys.path.append("../../")
from crawl.pdf2text_scipdf import extract_text_from_dict
from crawl.pdf2text_pypdf2 import split_into_paragraphs
np.random.seed(0)

def get_json_list(file_path, start_at=0, end_at=None):
    with open(file_path, "r") as f:
        json_list = []
        for idx, line in enumerate(f):
            if end_at is not None and idx >= end_at:
                return json_list
            elif idx < start_at:
                continue
            json_list.append(json.loads(line))
        return json_list
class ACLDataSet(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)    

class TrainSciBERT():
    def __init__(self, model=True):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model=None
        if model: 
            self.model = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')

    def build_training_corpus(self, input_dir, from_file=None, sample=None):
        """build continue-pretraining corpus

        Args:
            input_dir (str)
        """
        # input_dir: /home/jiaruil5/anlp/crawl/text/scipdf/json/
        # build paragraph list from all papers
        paragraphs = []
        listdir = os.listdir(input_dir)
        if from_file is not None:
            # print(len(listdir), listdir[0])
            tmp_listdir = [i.replace('.txt', '.json') for i in get_json_list(from_file)]
            listdir = [i for i in tmp_listdir if i in listdir]
            # print(len(tmp_listdir), tmp_listdir[0])
            print(len(listdir))
        if sample is not None:
            listdir = np.random.choice(listdir, sample, replace=False)

        for filename in listdir:
            json_path = os.path.join(input_dir, filename)
            article_dict = json.load(open(json_path, 'r'))
            text = split_into_paragraphs(extract_text_from_dict(article_dict))
            paragraphs.extend(text)
        
        # build bag list
        bags = []
        for paragraph in paragraphs:
            bags.extend(sent_tokenize(paragraph))
        bag_size = len(bags)
        
        # create 50 / 50 NSP (next sentence prediction) training data
        sentence_a, sentence_b, label = [], [], []
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            num_sentences = len(sentences)
            if num_sentences > 1:
                start = random.randint(0, num_sentences-2)
                # 50/50 whether is IsNextSentence or NotNextSentence
                if random.random() >= 0.5:
                    # this is IsNextSentence
                    sentence_a.append(sentences[start])
                    sentence_b.append(sentences[start+1])
                    label.append(0)
                else:
                    index = random.randint(0, bag_size-1)
                    # this is NotNextSentence
                    sentence_a.append(sentences[start])
                    sentence_b.append(bags[index])
                    label.append(1)
        
        return sentence_a, sentence_b, label
        
    def build_test_corpus(self, input_dir, from_file, sample=10):
        return self.build_training_corpus(input_dir, from_file, sample=sample)
    
    def build_dataset(self, sentence_a, sentence_b, label):
        inputs = self.tokenizer(
            sentence_a, 
            sentence_b, 
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length',
        )
        inputs['next_sentence_label'] = torch.LongTensor([label]).T
        inputs['labels'] = inputs.input_ids.detach().clone()
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)
        
        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103
        return inputs
    
    def train(self, inputs, out_dir, **kwargs):
        dataset = ACLDataSet(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=kwargs.get("batch_size", 16), shuffle=True)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        
        self.model.train()
        optim = AdamW(self.model.parameters(), lr=kwargs.get("lr", 5e-5))
        
        epochs = kwargs.get("epochs", 2)
        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                next_sentence_label = batch['next_sentence_label'].to(device)
                labels = batch['labels'].to(device)
                # process
                outputs = self.model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                next_sentence_label=next_sentence_label,
                                labels=labels)
                # extract loss
                loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
                # print(f'Epoch {epoch}')
                # print(loss.item())
            # model save
            filename = os.path.join(out_dir, f"lr_{str(kwargs.get('lr', 5e-5))}_bs_{str(kwargs.get('batch_size', 16))}_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), filename)


    def eval(self, inputs, model):
        dataset = ACLDataSet(inputs)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        
        model.eval()
        loss = []
        for batch in tqdm(loader, leave=True):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                next_sentence_label=next_sentence_label,
                                labels=labels)
            loss.append(outputs.loss.item())
            # print(outputs)
        print(np.mean(loss))
    
if __name__ == "__main__":
    cls = TrainSciBERT()
    
    a, b, l = cls.build_training_corpus("/home/jiaruil5/anlp/crawl/text/scipdf/json/")
    
    # df = pd.DataFrame()
    # df["sentence_a"] = a
    # df['sentence_b'] = b
    # df['label'] = l
    # df.to_csv("/home/jiaruil5/anlp/model/data/scibert_nsp.csv")

