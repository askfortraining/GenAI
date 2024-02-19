import pinecone
import streamlit as st
import os
from pinecone import Pinecone as Pinestore
#from langchain_community.llms.HuggingFaceHub import HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_community.vectorstores import Pinecone
# from langchain_pinecone import Pinecone
# from langchain.vectorstores import Pinecone
from langchain_pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

def load_doc(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

    # def embedding_docs(docs):
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # api_key = '2537f19d-cbf3-43df-8cbf-f5de949480a1'
    # pc = Pinestore(api_key=api_key)
    # index_name = "genaidemo"
    # index = pc.Index(index_name)
    # id = Pinecone.from_documents(docs,embeddings)
    #Index = Pinecone.from_documents(docs, embeddings, index_name='genaidemo')


def search_doc(query):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    doc_sec = Pinecone.from_existing_index(index_name='genaidemo', embedding=embeddings)
    similar_doc = doc_sec.similarity_search(query, k=2)
    llm = HuggingFaceHub(repo_id='bigscience/bloom', model_kwargs={"temperature":1e-10})
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=similar_doc, question=query)
    return response


if __name__ == '__main__':
    directory = 'C:/Doc/'
    documents = load_doc(directory)
    print(len(documents))
    docs = split_docs(documents)
    print(len(docs))
    # embedding_docs(docs)
    query = "What is acute pancreatitis?"
    search = search_doc(query)
    print(search)
