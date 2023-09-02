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

 
load_dotenv()
 

st.header("Annual Report Reader")

# upload a PDF file
pdf = st.file_uploader("Upload the Annual Report", type='pdf')
# pdf_sub = st.button("Submit AR")

if pdf is not None:
    print("Reading pdf")
    
    temp_path = f"temp_{pdf.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf.read())

    pdf_loader = PyPDFLoader(temp_path)
    documents = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    print("text read and splitted")

    
    persist_directory = 'db'
    embedding = OpenAIEmbeddings()
    print("//////////////////////////////")
    if os.path.exists('db'):
        print("inside if")
        # Now we can load the persisted database from disk, and use it as normal. 
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        print(f"{vectordb} Loaded ...")
    else:
        print("inside else")  
        # Embed and store the texts
        # Supplying a persist_directory will store the embeddings on disk
        vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
        vectordb.persist()
        # vectordb = None
        print(f"{vectordb} persisted ...")
        
    print("Start asking questions")

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")  
    submit_bt = st.button("Ask!")
    if query and submit_bt:
        print(query)
        docs = vectordb.similarity_search(query=query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)
    else:
        print("No query")
