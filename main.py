import os
from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Ludo Blitz GD Assistant")


promt = st.text_input("Question", placeholder="Enter your question here..")

if "user_promt_history" not in st.session_state:
    st.session_state["user_promt_history"] = []

if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def create_source_string(source_urls: set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source.rsplit('/', 1)[-1]}\n"
    return sources_string


if promt:
    with st.spinner("Thinking ..."):
        generated_response = run_llm(
            query=promt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['answer']} \n\n {create_source_string(sources)}"
        )

        st.session_state["user_promt_history"].append(promt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append((promt, generated_response["answer"]))

if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answer_history"], st.session_state["user_promt_history"]
    ):
        message(user_query, is_user=True)
        message(generated_response)
