from typing import Literal
import streamlit as st
from dotenv import load_dotenv
import os
import datetime
import json
import utils
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import openai

# https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/04-langchain-chat.ipynb

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, "../.env"))

st.set_page_config(
    page_title="Simplest Chat (LangChain)",
)

st.title("Simplest Chat (LangChain)")

with st.expander("Settings"):
    do_streaming = st.checkbox("Streaming output", True)
    show_metadata = st.checkbox("Show metadata", True)
    initial_system_message = st.text_area("System Message (modify this will clear history)", "How can I help you?")

with st.sidebar:
    openai_selection = utils.generate_api_and_language_model_selection()


if (
    "simplest_langchain_chat_messages" not in st.session_state
    or st.session_state.simplest_langchain_chat_messages
)[0].content != initial_system_message:
    st.session_state["simplest_langchain_chat_messages"] = [
        SystemMessage(content=initial_system_message),
    ]
    st.session_state["simplest_langchain_chat_metadata"] = [
        # TODO: cost of system message
        {}
    ]


download_button = st.empty()


# Render history messages
for msg in utils.convert_langchain_to_openai_message_history(
    st.session_state.simplest_langchain_chat_messages,
    st.session_state.simplest_langchain_chat_metadata,
):
    role = msg["role"]
    if msg["role"] == "error":
        role = "assistant"
    with st.chat_message(role):
        if msg["role"] != "error":
            st.write(msg["content"])
        else:
            st.error(msg["content"])

        if show_metadata:
            metadata = msg["metadata"]
            if "finish_reason" in metadata:
                st.caption(f"Finish reason: {metadata['finish_reason']}")

            if "error_reason" in metadata:
                st.caption(f"Error reason: {metadata['error_reason']}")

if prompt := st.chat_input():
    # TODO: maybe unify a ModelCreator for `client` creation
    if openai_selection == "OpenAI":
        if not st.session_state.openai_api_key:
            st.warning("🥸 Please add your OpenAI API key to continue.")
            st.stop()

        client = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            temperature=st.session_state.temperature,
            model=st.session_state.model,
        )

    elif openai_selection == "Azure OpenAI":
        if not st.session_state.azure_openai_api_key:
            st.warning("🥸 Please add your Azure OpenAI API key to continue.")
            st.stop()

        client = AzureChatOpenAI(
            api_key=st.session_state.azure_openai_api_key,
            azure_endpoint=st.session_state.azure_openai_endpoint,
            azure_deployment=st.session_state.azure_openai_deployment_name,
            api_version=st.session_state.azure_openai_version,
            temperature=st.session_state.temperature,
            model=st.session_state.model,
        )

    st.session_state.simplest_langchain_chat_messages.append(
        HumanMessage(content=prompt)
    )
    st.session_state.simplest_langchain_chat_metadata.append({})
    st.chat_message("user").write(prompt)

    if not do_streaming:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # TODO: try except error
            # TODO: add metadata
            try:
                msg = client(
                    [
                        msg
                        for msg in st.session_state.simplest_langchain_chat_messages
                        if msg.type != "error"
                    ]
                )
                st.session_state.simplest_langchain_chat_messages.append(
                    AIMessage(content=msg.content)
                )
                st.session_state.simplest_langchain_chat_metadata.append({})
                message_placeholder.write(msg.content)
            except openai.BadRequestError as e:
                (
                    error_message,
                    error_reason,
                ) = utils.extract_error_from_openai_BadRequestError(e)

                message_placeholder.error(error_message)

                st.session_state.simplest_langchain_chat_messages.append(
                    utils.ErrorMessage(content=error_message)
                )
                st.session_state.simplest_langchain_chat_metadata.append(
                    {"error_reason": error_reason}
                )
                if show_metadata:
                    st.caption(f"Error reason: {error_reason}")

    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                full_response = ""
                for chunk in client.stream(
                    [
                        msg
                        for msg in st.session_state.simplest_langchain_chat_messages
                        if msg.type != "error"
                    ]
                ):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                st.session_state.simplest_langchain_chat_messages.append(
                    AIMessage(content=full_response)
                )
                st.session_state.simplest_langchain_chat_metadata.append({})

            except openai.BadRequestError as e:
                (
                    error_message,
                    error_reason,
                ) = utils.extract_error_from_openai_BadRequestError(e)

                message_placeholder.error(error_message)

                st.session_state.simplest_langchain_chat_messages.append(
                    utils.ErrorMessage(content=error_message)
                )
                st.session_state.simplest_langchain_chat_metadata.append(
                    {"error_reason": error_reason}
                )
                if show_metadata:
                    st.caption(f"Error reason: {error_reason}")

# TODO: maybe summarize content for the file name
download_button.download_button(
    "Download current chat history",
    # TypeError: Object of type SystemMessage is not JSON serializable
    json.dumps(
        utils.convert_langchain_to_openai_message_history(
            st.session_state.simplest_langchain_chat_messages,
            st.session_state.simplest_langchain_chat_metadata,
        ),
        indent=4,
        ensure_ascii=False,
    ),
    f"history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
    mime="text/plain",
    disabled=not st.session_state.simplest_langchain_chat_messages,
)
