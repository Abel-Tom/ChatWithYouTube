import streamlit as st
import langchain_helper as lch
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.sidebar.text_area(
            label="What is the YouTube video URL?",
            max_chars=50
            )
        query = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=50,
            key="query"
            )
        openai_api_key = st.sidebar.text_input(
            label="OpenAI API Key",
            key="langchain_search_api_key_openai",
            type="password"
            )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        submit_button = st.form_submit_button(label='Submit')
if openai_api_key and openai_api_key.startswith('sk-'):
    assistant = lch.Assistant(openai_api_key)
    if youtube_url:
         db = assistant.create_db_from_youtube_video_url(youtube_url)
         if query:
            response, docs = assistant.get_response_from_query(db, query)
            st.subheader("Answer:")
            st.text(textwrap.fill(response, width=85))      
else:
    st.info("Please add your OpenAI API key to continue.")
            