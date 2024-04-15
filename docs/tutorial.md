# Building a local knowledge base
Here we wil learn to build a local collection of PDF files and instructing Libby to index them, that is, embdding them into a vector space so that they can be used for generative activities.

To get the PDF we will use the pyzotero library to extract the PDF liles from our zotero library. 



below we use bash to move all PDF files in a directory tree to a folder called `pdfs` and then we use the `pdf2txt.py` script from the `pdfminer` library to extract the text from the PDF files. 
```bash
find . -name "*.pdf" -exec cp {} pdfs \;
cd pdfs
for f in *.pdf; do pdf2txt.py $f > $f.txt; done
```

