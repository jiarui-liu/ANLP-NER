from transformers import AutoTokenizer, BertForPreTraining
import torch
import os
import json
import nltk
from nltk.tokenize import sent_tokenize
import sys
import random
import pandas as pd

sys.path.append("../../")
from crawl.pdf2text_scipdf import extract_text_from_dict
from crawl.pdf2text_pypdf2 import split_into_paragraphs

class TrainSciBERT():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = BertForPreTraining.from_pretrained('allenai/scibert_scivocab_uncased')
        
    def build_training_corpus(self, input_dir):
        """build continue-pretraining corpus

        Args:
            input_dir (str)
        """
        # input_dir: /home/jiaruil5/anlp/crawl/text/scipdf/json/
        # build paragraph list from all papers
        paragraphs = []
        for filename in os.listdir(input_dir):
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
        
    # def build_dataset(self, sentence_a, sentence_b, label):
        

if __name__ == "__main__":
    cls = TrainSciBERT()
    
    a, b, l = cls.build_training_corpus("/home/jiaruil5/anlp/crawl/text/scipdf/json/")
    df = pd.DataFrame()
    df["sentence_a"] = a
    df['sentence_b'] = b
    df['label'] = l
    df.to_csv("/home/jiaruil5/anlp/model/data/scibert_nsp.csv")

