import scipdf
import spacy
import os
import json
from crawl.pdf2text_pypdf2 import split_into_paragraphs, process_paragraphs

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

def extract_text_from_dict(article_dict):
    text = article_dict.get("title", "")
    if text != "":
        text += '.\n'
    text += 'Abstract '
    text += article_dict.get("abstract", "")
    for section in article_dict.get("sections", []):
        heading = section.get("heading", "")
        if heading != "":
            text += "\n" + heading + " " 
        txt = section.get("text", "")
        if txt != "":
            text += txt + " "
    return text

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    pdf_directory = './pdf/'
    output_directory = './text/scipdf/'
    err_file = "pdf2text.error.jsonl"
    err_f = open(err_file, 'a')
    err_json = [i['filename'] for i in get_json_list(err_file)]
    
    test = True
    if test:
        test_file = "test_continue_pretrain.jsonl"
        test_f = open(test_file, 'a')

    for i in [output_directory, output_directory + 'json/', output_directory + 'formatted/']:
        if not os.path.exists(i):
            os.makedirs(i)

    generated_files = os.listdir(output_directory + 'formatted/')
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf') and filename.replace(".pdf", ".txt") not in generated_files and filename not in err_json:
            pdf_path = os.path.join(pdf_directory, filename)
            try:
                article_dict = scipdf.parse_pdf_to_dict(pdf_path) # return dictionary
                json_filename = os.path.splitext(filename)[0] + '.json'
                json_path = os.path.join(output_directory + 'json/', json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as txt_file:
                    json.dump(article_dict, txt_file)
                
                text = extract_text_from_dict(article_dict)
                paragraphs = split_into_paragraphs(text)

                processed_text = process_paragraphs(paragraphs, nlp)
                
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(output_directory + 'formatted/', txt_filename)

                print(f"Processed '{filename}' and saved as '{txt_filename}'.")
                
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(processed_text)
                print(filename)
                if test:
                    json.dump(filename.replace(".pdf", ".txt"), test_f)
                    test_f.write("\n")
                    
            except Exception as e:
                json.dump({'filename': filename}, err_f)
                err_f.write("\n")
        
        # break