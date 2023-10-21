f = "/home/jiaruil5/anlp/crawl/bib2pdf.jsonl"
out_f = open("/home/jiaruil5/anlp/crawl/bib2pdf_.jsonl", "a")
cnt = open(f, 'r').read()


cnt_list = cnt.split(""".pdf"}""")
for i in cnt_list:
    if i != "":
        out_f.write(i + """.pdf"}""" + "\n")