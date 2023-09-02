import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import time 

 
load_dotenv()

def main():
    st.header("Annual Report Reader")

    pdf = st.file_uploader("Upload the Annual Report", type='pdf')

    if pdf is not None:
        temp_path = f"temp_{pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(pdf.read())

        pdf_loader = PyPDFLoader(temp_path)
        documents = pdf_loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)

        # Create the embeddings
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(documents, embedding=embeddings, persist_directory='./data')

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = vector_store.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)
        else:
            print("No query")

if __name__ == '__main__':
    main()
