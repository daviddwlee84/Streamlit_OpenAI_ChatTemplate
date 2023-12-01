import streamlit as st
from dotenv import load_dotenv
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(curr_dir, ".env"))

st.set_page_config(
    page_title="Streamlit OpenAI Chat Demo",
    page_icon="👋",
)

st.write("# Streamlit OpenAI Chat Demo")

with st.sidebar:
    st.markdown(
        """
    ## LLM API Keys

    - [Get an OpenAI API key](https://platform.openai.com/account/api-keys)

    Default key is loaded from `.env`.
    You can fill keys here or later.
    """
    )

    st.text("OpenAI")
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
        type="password",
    )

    st.divider()

    st.text("Azure OpenAI")
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

st.markdown(
    """
    - Github Page: [daviddwlee84/Streamlit_OpenAI_ChatTemplate: Simple chat App template using Streamlit Chat component with OpenAI Key. Which is useful as a starter template to build a chat app.](https://github.com/daviddwlee84/Streamlit_OpenAI_ChatTemplate)
    - Personal Website: [David Lee](https://dwlee-personal-website.netlify.app/)
"""
)
