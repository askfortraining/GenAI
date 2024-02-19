import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone

def load_doc(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs



if __name__ == '__main__':
    directory = 'C:/Doc/'
    documents = load_doc(directory)
    print(len(documents))
    docs = split_docs(documents)
    print(len(docs))
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    query_result = embeddings.embed_query("hello buddy")
    print(len(query_result))
