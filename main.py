import streamlit as st
import os
from langchain_community.llms import HuggingFaceHub

# Press the green button in the gutter to run the script.
st.set_page_config(page_title="Demo1", page_icon=":robot:")
st.header("Genn AI - Learning")


def get_text():
    input_text = st.text_input("You: ", key="input")
    if input_text == "":
        user_input = "hi"
    else:
        user_input = input_text
    return user_input


def load_answer(question):
    repo_id = "google/flan-t5-xxl"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
    )
    answer = llm(question)
    return answer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    user_response = get_text()
    response = load_answer(user_response)
    submit = st.button('Generate')
    if submit:
        st.write(response)
