import streamlit as st
from dotenv import load_dotenv
import os
import openai

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, "../.env"))

st.set_page_config(
    page_title="Simplest Chat",
)

st.title("Simplest Chat")

"""
- Every chat are independent i.e. we don't pass the chat history to ChatGPT
"""

with st.sidebar:
    openai_selection = st.selectbox("OpenAI Version", ["OpenAI", "Azure OpenAI"])

    if openai_selection == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
            type="password",
        )
    elif openai_selection == "Azure OpenAI":
        st.session_state.azure_openai_api_key = st.text_input(
            "Azure OpenAI API Key",
            value=st.session_state.get(
                "azure_openai_api_key", os.getenv("AZURE_OPENAI_KEY")
            ),
            type="password",
        )
        st.session_state.azure_openai_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            value=st.session_state.get(
                "azure_openai_endpoint", os.getenv("AZURE_OPENAI_ENDPOINT")
            ),
            type="default",
        )
        st.session_state.azure_openai_deployment_name = st.text_input(
            "Azure OpenAI Deployment Name",
            value=st.session_state.get(
                "azure_openai_deployment_name",
                os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            ),
            type="default",
        )
        st.session_state.azure_openai_version = st.text_input(
            "Azure OpenAI Version",
            value=st.session_state.get(
                "azure_openai_version", os.getenv("AZURE_OPENAI_VERSION")
            ),
            type="default",
        )

if "simplest_chat_messages" not in st.session_state:
    st.session_state["simplest_chat_messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.simplest_chat_messages:
    st.chat_message(msg["role"]).write(msg["content"])

# https://streamlit.io/generative-ai
# TODO: make response streaming https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
if prompt := st.chat_input():
    if openai_selection == "OpenAI":
        if not st.session_state.openai_api_key:
            st.warning("🥸 Please add your OpenAI API key to continue.")
            st.stop()

        client = openai.OpenAI(api_key=st.session_state.openai_api_key)

    elif openai_selection == "Azure OpenAI":
        if not st.session_state.azure_openai_api_key:
            st.warning("🥸 Please add your Azure OpenAI API key to continue.")
            st.stop()

        client = openai.AzureOpenAI(
            api_key=st.session_state.azure_openai_api_key,
            azure_endpoint=st.session_state.azure_openai_endpoint,
            azure_deployment=st.session_state.azure_openai_deployment_name,
            api_version=st.session_state.azure_openai_version,
        )

    st.session_state.simplest_chat_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=st.session_state.simplest_chat_messages
    )
    msg = response.choices[0].message
    st.session_state.simplest_chat_messages.append(
        {"role": "assistant", "content": msg.content}
    )
    st.chat_message("assistant").write(msg.content)
