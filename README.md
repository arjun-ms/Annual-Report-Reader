![Chroma Frame](https://github.com/TH-Activities/saturday-hack-night-template/assets/90635335/365c00da-597c-446f-9aa7-bed99fb26074)

# Annual Report Reader
Our project aims to create an interactive and user-friendly web application that allows users to extract insights and information from Annual Reports of a company PDF documents. 
We integrate Langchain, OpenAI, and ChromaDB to deliver accurate and relevant responses.

## Team members
1. [Arjun M S](https://github.com/arjun-ms)
2. [Muhammed Ajmal](https://github.com/ajmalmohad)
3. [Namitha S](https://github.com/Namitha-S-11465)

## Link to product walkthrough
   [link to video](Link Here)

## How it Works ?
This web application harnesses the power of Langchain, OpenAI, and ChromaDB to facilitate efficient PDF document interaction. 
Users upload PDFs, which are automatically processed into smaller text chunks for analysis. 
OpenAI's language model is employed to handle user questions with precision, providing contextually relevant answers. 
ChromaDB stores document embeddings, enabling retrieval of information from the PDFs. 
Users simply input their queries, and the system matches these inquiries with the PDF content, offering concise and accurate responses.

## Libraries used
1. Chromadb
2. Langchain
3. OpenAI
4. PyPDF
   
## How to configure
1. Clone this repository
```
$git clone https://github.com/arjun-ms/Annual-Report-Reader.git
```
2. Install the libraries given in requirements.txt
```
$pip install -r requirements.txt
```
   
## How to Run
```
$streamlit run annualreport.py
```
