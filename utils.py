import streamlit as st
from typing import Literal, List, Dict, Tuple
import os
import tiktoken
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
import openai


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


def num_tokens_from_messages(messages: str, model: str = "gpt-3.5-turbo-0613") -> int:
    """
    TODO: haven't used yet (this is for calculating cost)

    Return the number of tokens used by a list of messages.
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


class ErrorMessage(BaseMessage):
    """A Message for priming AI behavior, usually passed in as the first of a sequence
    of input messages.
    """

    type: Literal["error"] = "error"


def convert_langchain_to_openai_message_history(
    langchain_chat_history: List[BaseMessage],
    metadata: List[dict] = [],
    include_error: bool = True,
) -> List[Dict[str, str]]:
    openai_chat_history = []

    if metadata:
        assert len(metadata) == len(langchain_chat_history)
    else:
        metadata = [None] * len(langchain_chat_history)

    for msg, meta in zip(langchain_chat_history, metadata):
        if msg.type == "human":
            if meta is None:
                openai_chat_history.append({"role": "user", "content": msg.content})
            else:
                openai_chat_history.append(
                    {"role": "user", "content": msg.content, "metadata": meta}
                )
        elif msg.type in {"system", "ai"}:
            if meta is None:
                openai_chat_history.append(
                    {"role": "assistant", "content": msg.content}
                )
            else:
                openai_chat_history.append(
                    {"role": "assistant", "content": msg.content, "metadata": meta}
                )
        elif msg.type == "error" and include_error:
            if meta is None:
                openai_chat_history.append(
                    {"role": "error", "content": msg.content}
                )
            else:
                openai_chat_history.append(
                    {"role": "error", "content": msg.content, "metadata": meta}
                )
        else:
            print('Found unknown message', msg)

    return openai_chat_history


def extract_error_from_openai_BadRequestError(
    error: openai.BadRequestError,
) -> Tuple[str, dict]:
    error = error.response.json()
    error_message = error["error"]["message"]
    error_reason = {}
    for reason, result in error["error"]["innererror"]["content_filter_result"].items():
        if result["filtered"]:
            error_reason[reason] = result["severity"]
    return error_message, error_reason
