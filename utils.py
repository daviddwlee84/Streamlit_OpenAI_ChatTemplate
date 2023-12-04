import streamlit as st
from typing import Literal
import os


def generate_api_and_language_model_selection() -> Literal["OpenAI", "Azure OpenAI"]:
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

    st.divider()

    st.session_state.model = st.text_input(
        "Model Name",
        value=st.session_state.get("model", "gpt-3.5-turbo"),
        type="default",
    )
    st.session_state.temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        step=0.1,
        value=st.session_state.get("temperature", 0.0),
    )

    return openai_selection
