1. Get all .bib files
    - ACL 2022, ACL 2023, EMNLP 2022, NAACL 2022
        ```bash
        wget https://aclanthology.org/events/acl-2022/#2022acl-short
        mv index.html acl-2022.html
        ```
    - use beautifulsoup to get .bib files: `html2bib.py`