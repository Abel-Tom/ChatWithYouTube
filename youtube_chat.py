import streamlit as st
import langchain_helper as lch

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    url = st.text_input("Enter the YouTube video url", key="url", type="default")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/TomHellCat/ChatWithYouTube)"

st.title("💬 Chat with YouTube")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not url and not url.startswith('https://www.youtube.com/watch?v='):
        st.info("Please enter a YouTube Video url to continue.")
        st.stop()
    if not url.startswith('https://www.youtube.com/watch?v='):
        st.info("Please enter a valid YouTube Video url to continue.")
        st.stop()
    assistant = lch.Assistant(openai_api_key)
    if url:
         db = assistant.create_db_from_youtube_video_url(url)
         if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response, docs = assistant.get_response_from_query(db, prompt) 
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)