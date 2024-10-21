import streamlit as st
from dotenv import load_dotenv
import os
import datetime
import json
import requests
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from functools import partial
import nest_asyncio

nest_asyncio.apply()

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, "../.env"))

st.set_page_config(
    page_title="RAG: Retrieval Augmented Generation (LightRAG)",
)

st.title("RAG: Retrieval Augmented Generation (LightRAG)")


with st.expander("Settings"):
    show_metadata = st.checkbox("Show metadata", False, disabled=True, help="TBD")
    working_dir_name = st.text_input(
        "LightRAG working directory name",
        "Temp",
        help="You would like to choose different directory for different retrieval database.",
    )
    working_dir = os.path.join(curr_dir, "../LightRAG_WorkingRoot", working_dir_name)
    os.makedirs(working_dir, exist_ok=True)

if os.listdir(working_dir):
    st.info(
        f"Working Directory {working_dir} is not empty (might have previous build result)."
    )

with st.sidebar:
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
        type="password",
    )
    mode_type = st.selectbox("Model", ["GPT4o-mini", "GPT4o"])

if not st.session_state.openai_api_key or not os.getenv("OPENAI_API_KEY"):
    st.warning("🥸 Please add your OpenAI API key to continue.")
    st.stop()

# UnicodeDecodeError: 'charmap' codec can't decode byte 0x81 in position 19799: character maps to <undefined>
rag = LightRAG(
    working_dir=(
        working_dir
        if working_dir
        else lambda: f"./lightrag_cache_{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    ),
    # KeyError: 'Could not automatically map gpt-4o-mini to a tokeniser. Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect.'
    # You need to upgrade tiktoken if you have this error
    llm_model_func=partial(
        (
            gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
            if mode_type == "GPT4o-mini"
            else gpt_4o_complete  # Optionally, use a stronger model
        ),
        api_key=st.session_state.openai_api_key,
    ),
    # llm_model_func=(
    #     gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    #     if mode_type == "GPT4o-mini"
    #     else gpt_4o_complete  # Optionally, use a stronger model
    # ),
)

st.session_state["lightrag_model"] = rag


@st.cache_data(show_spinner=f"Parsing URL by [Jina-AI Reader](https://jina.ai/reader/)")
def parse_url_with_jinja_ai_reader(url: str) -> str:
    return requests.get(f"https://r.jina.ai/{url}").content.decode()


@st.fragment
def upload_content():
    content_type = st.selectbox(
        "Content Type",
        ["Plain Text", "URL"],
        help="TODO: include upload txt file or more type with parser",
        index=1,
    )
    if content_type == "Plain Text":
        content = st.text_area("Content")
    elif content_type == "URL":
        url = st.text_input(
            "URL",
            help="If you don't know what to test, you can use LightRAG paper `https://arxiv.org/html/2410.05779v1`.",
        )
        with st.expander("Parsed Content", expanded=bool(url)):
            if url:
                content = parse_url_with_jinja_ai_reader(url)
                st.markdown(content)
            else:
                st.warning("No URL given...")

    if st.button("Insert Knowledge!", type="primary"):
        with st.spinner("Inserting Knowledge..."):
            st.session_state["lightrag_model"].insert(content)
        st.text("done")


insert_knowledge_tab, chat_tab = st.tabs(["Insert Knowledge", "Chat"])
with insert_knowledge_tab:
    upload_content()

if f"lightrag_chat_history_{working_dir_name}" not in st.session_state:
    st.session_state[f"lightrag_chat_history_{working_dir_name}"] = []


@st.fragment
def chat():
    download_button = st.empty()

    # Render history messages
    for msg in st.session_state[f"lightrag_chat_history_{working_dir_name}"]:
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

    query_mode = st.selectbox(
        "Query Mode", ["naive", "local", "global", "hybrid"], index=3
    )
    response_type: str = st.text_input(
        "Response Type",
        "Multiple Paragraphs",
        help="Hint for LLM to reply in which format. Author's default is `Multiple Paragraphs`",
    )
    if prompt := st.chat_input():
        # TODO: able to show LightRAG intermediate metadata
        st.session_state[f"lightrag_chat_history_{working_dir_name}"].append(
            {"role": "user", "content": prompt, "metadata": {}}
        )
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                response = rag.query(
                    prompt,
                    param=QueryParam(mode=query_mode, response_type=response_type),
                )
            st.session_state[f"lightrag_chat_history_{working_dir_name}"].append(
                {"role": "assistant", "content": response, "metadata": {}}
            )
            message_placeholder.write(response)

    # TODO: maybe summarize content for the file name
    download_button.download_button(
        "Download current chat history",
        json.dumps(
            st.session_state[f"lightrag_chat_history_{working_dir_name}"],
            indent=4,
            ensure_ascii=False,
        ),
        f"history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
        mime="text/plain",
        disabled=not st.session_state[f"lightrag_chat_history_{working_dir_name}"],
    )


with chat_tab:
    chat()
