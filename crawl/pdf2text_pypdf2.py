import os
from PyPDF2 import PdfReader
import re
import spacy

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        num_pages = len(pdf.pages)

        text_content = ""

        for i in range(num_pages):
            text = pdf.pages[i].extract_text()
            if text:
                text_content += text
        return text_content

def split_into_paragraphs(text):
    paragraphs = re.split(r'(?<=[\.\?\!])\n', text)
    return paragraphs

def process_paragraphs(paragraphs, nlp):
    processed_text = ""
    for paragraph in paragraphs:
        if paragraph.strip():  
            paragraph = paragraph.replace('\n','')
            paragraph = paragraph.encode('utf-8', errors='ignore').decode('utf-8')
            doc = nlp(paragraph)
            tokens = [token.text for token in doc]
            paragraph_utf8 = ' '.join(tokens).encode('utf-8', errors='ignore').decode('utf-8')
            processed_text += paragraph_utf8 + "\n"
    return processed_text

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")

    pdf_directory = './pdf'

    output_directory = './text'

    processed_files_log = './processed.txt'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    processed_set = set()

    with open(processed_files_log, 'r', encoding='utf-8') as file:
        for line in file:
            processed_set.add(line.strip())

    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            if filename in processed_set: 
                print('done: ', filename)
                continue

            pdf_path = os.path.join(pdf_directory, filename)
            
            text = extract_text_from_pdf(pdf_path)
            paragraphs = split_into_paragraphs(text)

            processed_text = process_paragraphs(paragraphs, nlp)

            txt_filename = os.path.splitext(filename)[0] + '.txt'

            txt_path = os.path.join(output_directory, txt_filename)

            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(processed_text)

            print(f"Processed '{filename}' and saved as '{txt_filename}'.")

            with open(processed_files_log, 'a', encoding='utf-8') as log:
                log.write(filename+ '\n')


    print("Processing complete.")
