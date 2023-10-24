Data (shared on Google Drive, [https://drive.google.com/drive/folders/1_vPJgkUscuTOhaH_5BuPbnHzomBwMZAg?usp=sharing](https://drive.google.com/drive/folders/1_vPJgkUscuTOhaH_5BuPbnHzomBwMZAg?usp=sharing)):
1. PDF data can be accessed in the subfolder `pdf/`
    - Right now it contains the following bib files, in total **3045** files:
    ```text
    "2023.acl-long.bib",
    "2023.acl-short.bib",
    "2022.emnlp-main.bib",
    "2022.naacl-main.bib",
    "2022.acl-long.bib",
    "2022.acl-short.bib",
    ```
2. BIB data can be accessed in the subfolder `bib/`
    - Including all bib files from ACL 2022, ACL 2023, EMNLP 2022, NAACL 2022

Directory structure of this repo:
```text
crawl/
├── html/
├── bib/
├── pdf/
├── src/
├── html2bib.py
├── bib2pdf.py
├── conf2bib.jsonl
├── bib2pdf.jsonl
└── bib2pdf.error.jsonl
```

Crawling steps:
1. Get all .bib files
    - ACL 2022, ACL 2023, EMNLP 2022, NAACL 2022
        ```bash
        wget https://aclanthology.org/events/acl-2022/#2022acl-short
        mv index.html acl-2022.html
        ```
    - use beautifulsoup to get .bib files: `html2bib.py`

2. Download all .pdf files: `bib2pdf.py`
3. Convert .pdf files to plain text files
    - Install spacy model version: `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.6.0/en_core_web_lg-3.6.0.tar.gz`
    - Follow the guide here to convert selected text to the desired format: [https://github.com/cmu-anlp/nlp-from-scratch-assignment-2023/blob/main/annotation_standard.md](https://github.com/cmu-anlp/nlp-from-scratch-assignment-2023/blob/main/annotation_standard.md)
    - Method 1: PyPDF2 `crawl/pdf2text_pypdf2.py`
    - Method 2: SciPDF, better than PyPDF2, can filter out tables, figures, and references, `crawl/pdf2text_scipdf.py`
        - After running this file, there are 3000 / 7942 pdf files run into convertion error, so their txt files only contain part of the converted text, error files see `pdf2text.error.jsonl`

- SciBERT: https://github.com/allenai/scibert
    - Continue-pretrain BERT: `model/sibert/continue_pretrain.py`, https://towardsdatascience.com/how-to-train-bert-aaad00533168
    - Fine-tune SciBERT: https://github.com/Nikoschenk/language_model_finetuning/blob/master/scibert_fine_tuner.ipynb, https://guptayash2010.medium.com/text-classification-with-scibert-a285d2f2db06