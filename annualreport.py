import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from constants import CHROMA_SETTINGS
from ingest import loadAndSplitDoc

load_dotenv()

# Initialize session state
if 'query' not in st.session_state:
    st.session_state.query = ""

# Initialize vectordb
vectordb = None

st.header("Annual Report Reader")

# Upload a PDF file
pdf = st.file_uploader("Upload the Annual Report", type='pdf')

# Accept user questions/query
st.session_state.query = st.text_input("Ask questions about your PDF file:", st.session_state.query)

if st.button("Submit AR and Ask"):
    if pdf is not None:
        temp_path = f"temp/temp_{pdf.name}"
        if not os.path.exists(temp_path):
            with open(temp_path, "wb") as f:
                f.write(pdf.read())
        
        persist_directory = 'chroma'
        embedding = OpenAIEmbeddings()
        
        if os.path.exists('chroma'):
            # Now we can load the persisted database from disk and use it as normal.
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, client_settings=CHROMA_SETTINGS)
            print(f"{vectordb} Loaded ...")
        else:
            documents = loadAndSplitDoc(temp_path)
            # Embed and store the texts
            # Supplying a persist_directory will store the embeddings on disk
            vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
            vectordb.persist()
            print(f"{vectordb} persisted ...")
        
        print("Start asking questions")

        
        # st.write(st.session_state.query)
        docs = vectordb.similarity_search(query=st.session_state.query, k=3)
        
        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=st.session_state.query)
            st.write(response)
            print(cb)

