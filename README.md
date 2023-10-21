Data (shared on Google Drive, [https://drive.google.com/drive/folders/1_vPJgkUscuTOhaH_5BuPbnHzomBwMZAg?usp=sharing](https://drive.google.com/drive/folders/
1_vPJgkUscuTOhaH_5BuPbnHzomBwMZAg?usp=sharing)):
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
    - `