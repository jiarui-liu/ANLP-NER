import scipdf
import spacy
import os
import json
from pdf2text_pypdf2 import split_into_paragraphs, process_paragraphs

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

    for i in [output_directory, output_directory + 'json/', output_directory + 'formatted/']:
        if not os.path.exists(i):
            os.makedirs(i)

    generated_files = os.listdir(output_directory)
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf') and filename.replace(".pdf", ".json") not in generated_files:
            pdf_path = os.path.join(pdf_directory, filename)
            try:
                article_dict = scipdf.parse_pdf_to_dict(pdf_path) # return dictionary
            except Exception as e:
                json.dump({'filename': filename}, err_f)
                err_f.write("\n")
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
        
        # break