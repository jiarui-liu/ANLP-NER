from bs4 import BeautifulSoup
import glob
import requests
import time
import os
import json

html_files = glob.glob("html/*.html")
print(html_files)
bib_dir = "bib/"

def download_file(url):
    local_filename = bib_dir + url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        if os.path.exists(local_filename):
            os.mknod(local_filename)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

f = open("conf2bib.jsonl", 'a')
for html_file in html_files:
    with open(html_file, 'r') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
        
        ul = soup.find_all("ul", attrs={'class': 'list-pl-responsive'})[-1]
        bib_list = ul.find_all("a", attrs={'class': 'align-middle'})
        # url_list = []
        for bib_a in bib_list:
            url_part = str(bib_a.get("href")).replace("#2022", "2022.").replace("#2023", "2023.") + '.bib'
            url = "https://aclanthology.org/volumes/" + url_part
            print(url)
            # url_list.append(url)
            dump_info = {
                "conference": html_file.split("/")[-1],
                "bib_url": url_part,
            }
            json.dump(dump_info, f, ensure_ascii=False)
            f.write("\n")
            # download_file(url)
            # time.sleep(1)
            
            