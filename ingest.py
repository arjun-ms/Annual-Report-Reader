import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import time 
from constants import CHROMA_SETTINGS

persist_directory = "db"

# load_dotenv()


# def loadAndSplitDoc(temp_path):
#     print("inside load and split")
#     pdf_loader = PyPDFLoader(temp_path)
#     documents = pdf_loader.load()

    
#     documents = text_splitter.split_documents(documents)

    

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    print("splitting into chunks")
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="openai-gpt")
    #create vector store here
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 

    print("text read and splitted")
    return documents

if __name__ == "__main__":
    main()
