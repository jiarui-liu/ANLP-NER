import bibtexparser
import os
import json
import requests
import time

bib_dir = "bib/"
pdf_dir = "pdf/"
f_list = [
    "2023.acl-long.bib",
    "2023.acl-short.bib",
    "2022.emnlp-main.bib",
    "2022.naacl-main.bib",
    "2022.acl-long.bib",
    "2022.acl-short.bib",
]

json_info_f = open("bib2pdf.jsonl", "a")
error_f = open("bib2pdf.error.jsonl", "a")

def download_pdf(url, pdf_path, max_tries=5):
    curr_tries = 0
    while curr_tries < max_tries:
        try: 
            r = requests.get(url, timeout=1, verify=True) 
            r.raise_for_status()
            with open(pdf_path, 'wb') as f:
                f.write(r.content)
            break
        except requests.exceptions.HTTPError as errh: 
            print("HTTP Error", errh) 
            print(errh.args[0]) 
        except requests.exceptions.ReadTimeout as errrt: 
            print("Time out", errrt) 
        except requests.exceptions.ConnectionError as conerr: 
            print("Connection error", conerr) 
        except requests.exceptions.RequestException as errex: 
            print("Exception request", errex) 
        except Exception as e:
            print(e)
        curr_tries += 1
        time.sleep(5)
    if curr_tries == max_tries:
        raise RuntimeError(f"Crawled error for pdf {url}")

def bib2pdf(file):
    lib = bibtexparser.parse_file(path=file)
    is_first=True
    for k, v in lib.entries_dict.items():
        # skip the first url
        if is_first:
            is_first=False
            continue
        # get info
        json_info = {}
        json_info['entry'] = k
        json_info['file'] = file.split("/")[-1]
        pdf_filename = k + '.pdf'
        pdf_filepath = pdf_dir + pdf_filename
        print(f"Crawling entry {k}")
        for field in v.fields:
            if field._key == "url":
                json_info['url'] = field._value + ".pdf"
            if field._key == 'title':
                json_info['title'] = field._value
            if field._key == 'booktitle':
                json_info['booktitle'] = field._value
        # crawl pdf
        if not os.path.isfile(pdf_filepath):
            try:
                download_pdf(json_info['url'], pdf_filepath)
                json.dump(json_info, json_info_f, ensure_ascii=False)
                json_info_f.write("\n")
                print(f"Crawling successfully {k}")
            except RuntimeError:
                json.dump(json_info, error_f, ensure_ascii=False)
                error_f.write("\n")
                print(f"Crawling error {k}")

if __name__ == "__main__":
    for f in f_list:
        bib2pdf(bib_dir + f)