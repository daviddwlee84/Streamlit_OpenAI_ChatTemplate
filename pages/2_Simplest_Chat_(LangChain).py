from typing import List
import streamlit as st
from dotenv import load_dotenv
import os
import datetime
import json
import utils
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/04-langchain-chat.ipynb

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, "../.env"))

st.set_page_config(
    page_title="Simplest Chat (LangChain)",
)

st.title("Simplest Chat (LangChain)")


with st.expander("Settings"):
    do_streaming = st.checkbox("Streaming output", False)
    show_metadata = st.checkbox("Show metadata", True)

with st.sidebar:
    openai_selection = utils.generate_api_and_language_model_selection()


if "simplest_langchain_chat_messages" not in st.session_state:
    # st.session_state["simplest_langchain_chat_messages"] = [
    #     # TODO: able to set system initial prompt (If reset prompt then clear history)
    #     {
    #         "role": "assistant",
    #         "content": SystemMessage(content="How can I help you?"),
    #         "metadata": {},
    #     }
    # ]
    st.session_state["simplest_langchain_chat_messages"] = [
        # TODO: able to set system initial prompt (If reset prompt then clear history)
        SystemMessage(content="How can I help you?"),
    ]
    st.session_state["simplest_langchain_chat_metadata"] = [
        # TODO: cost of system message
        {}
    ]


# def _get_langchain_messages(chat_history: List[dict]) -> List[BaseMessage]:
#     return [message["content"] for message in chat_history]

openai_message_history = utils.convert_langchain_to_openai_message_history(
    st.session_state.simplest_langchain_chat_messages,
    st.session_state.simplest_langchain_chat_metadata,
)

# TODO: maybe summarize content for the file name
st.download_button(
    "Download current chat history",
    # TypeError: Object of type SystemMessage is not JSON serializable
    json.dumps(
        openai_message_history,
        indent=4,
        ensure_ascii=False,
    ),
    f"history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
    mime="text/plain",
    disabled=not openai_message_history,
)

# Render history messages
for msg in openai_message_history:
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

# https://streamlit.io/generative-ai
# TODO: make response streaming https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
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
            msg = client(st.session_state.simplest_langchain_chat_messages)
            st.session_state.simplest_langchain_chat_messages.append(
                AIMessage(content=msg.content)
            )
            st.session_state.simplest_langchain_chat_metadata.append({})

            message_placeholder.write(msg.content)
    else:
        raise NotImplementedError()
