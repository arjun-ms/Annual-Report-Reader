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

 
load_dotenv()


def loadAndSplitDoc(temp_path):
    print("inside load and split")
    pdf_loader = PyPDFLoader(temp_path)
    documents = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    print("text read and splitted")
    return documents
