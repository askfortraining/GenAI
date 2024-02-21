import pinecone
import streamlit as st
import os
from pinecone import Pinecone as Pinestore
# from langchain_community.llms.HuggingFaceHub import HuggingFaceHub
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

st.set_page_config(page_title="Demo1", page_icon=":robot:")
st.header("Genn AI - Learning")


def get_text():
    input_text = st.text_input("You: ", key="input")
    if input_text == "":
        user_input = "hi"
    else:
        user_input = input_text
    return user_input


def search_doc(query):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    doc_sec = Pinecone.from_existing_index(index_name='genaidemo', embedding=embeddings)
    similar_doc = doc_sec.similarity_search(query, k=1)
    llm = HuggingFaceHub(repo_id='bigscience/bloom', model_kwargs={"temperature":1.0, "max_length":2048})
    chain = load_qa_chain(llm, chain_type="stuff",verbose= True)
    response = chain.run(input_documents=similar_doc, question=query)
    return response


if __name__ == '__main__':
    query = get_text()
    search = search_doc(query)
    submit = st.button('Generate')
    if submit:
        st.write(search)
