from typing import List
import streamlit as st
from dotenv import load_dotenv
import os
import openai
import datetime
import json
import utils

# from openai import OpenAI, OpenAIError
import googlesearch

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, "../.env"))

st.set_page_config(
    page_title="With Search Engine (OpenAI)",
)

st.title("With Search Engine (OpenAI)")

st.caption("We are using Traditional Chinese in this example.")


with st.expander("Settings"):
    search_num_result = st.number_input("Search Result", value=3)
    do_streaming = st.checkbox("Streaming output", True)
    show_metadata = st.checkbox("Show metadata", True)
    initial_system_message = st.text_area(
        "System Message (modify this will clear history)", "小助理"
    )

with st.sidebar:
    openai_selection = utils.generate_api_and_language_model_selection()

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


# def get_reply(messages: List[Dict[str, str]]) -> str:
#     try:
#         response = client.chat.completions.create(
#             model=st.session_state.model, messages=messages
#         )
#         reply = response.choices[0].message.content
#     except OpenAIError as err:
#         reply = f"Getting {err.error.type} error\n{err.error.message}"
#     return reply

if (
    "with_search_engine_chat_messages" not in st.session_state
    or st.session_state.with_search_engine_chat_messages[0]["content"]
    != initial_system_message
):
    st.session_state["with_search_engine_chat_messages"] = [
        # TODO: able to set system initial prompt (If reset prompt then clear history)
        {"role": "assistant", "content": initial_system_message, "metadata": {}}
    ]


download_button = st.empty()

# Render history messages
for msg in st.session_state.with_search_engine_chat_messages:
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


@st.cache_data(show_spinner="Search Google...")
def search_google(text: str) -> List[googlesearch.SearchResult]:
    return list(googlesearch.search(
        text, advanced=True, num_results=search_num_result, lang="zh-TW"
    ))


# https://streamlit.io/generative-ai
# TODO: make response streaming https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
col1, col2 = st.columns(2)
search_web_first = col1.checkbox(
    "Search Web First", True, help="Search web based on current input"
)
reply_latest_history = col2.checkbox(
    "Reply Latest History", False, help="Reply only based on latest history (no memory)"
)
if prompt := st.chat_input():
    # TODO: maybe unify a ModelCreator for `client` creation
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

    st.session_state.with_search_engine_chat_messages.append(
        {"role": "user", "content": prompt, "metadata": {}}
    )

    if search_web_first:
        search_result = "以下為已經發生的事實：\n\n"
        for res in search_google(prompt):
            search_result += f"標題：{res.title}\n" f"摘要：{res.description}\n\n"
        search_result += "請依照上述事實回答問題 \n"
        st.session_state.with_search_engine_chat_messages.append(
            {"role": "user", "content": search_result, "metadata": {}}
        )
        st.chat_message("user").write(search_result)
    st.chat_message("user").write(prompt)

    history_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.with_search_engine_chat_messages
        if m["role"] in {"user", "assistant"}
    ]
    if reply_latest_history:
        if search_result:
            history_messages = history_messages[-2:]
        else:
            history_messages = history_messages[-1:]

    if not do_streaming:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = client.chat.completions.create(
                    model=st.session_state.model,
                    # Chat history
                    # TODO: don't send user message cause error
                    messages=history_messages,
                    temperature=st.session_state.temperature,
                    stream=False,
                )
                msg = response.choices[0].message
                st.session_state.with_search_engine_chat_messages.append(
                    {"role": "assistant", "content": msg.content, "metadata": {}}
                )
                message_placeholder.write(msg.content)

            except openai.BadRequestError as e:
                error = e.response.json()
                error_message = error["error"]["message"]
                error_reason = {}
                for reason, result in error["error"]["innererror"][
                    "content_filter_result"
                ].items():
                    if result["filtered"]:
                        error_reason[reason] = result["severity"]

                message_placeholder.error(error_message)

                st.session_state.with_search_engine_chat_messages.append(
                    {
                        "role": "error",
                        "content": error_message,
                        "metadata": {"error_reason": error_reason},
                    }
                )
                if show_metadata:
                    st.caption(f"Error reason: {error_reason}")

    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                full_response = ""
                for response in client.chat.completions.create(
                    model=st.session_state.model,
                    # NOTE: this can prevent from additional messages
                    # TODO: don't send user message cause error
                    messages=history_messages,
                    temperature=st.session_state.temperature,
                    stream=True,
                ):
                    full_response += (
                        # NOTE: response.choices[0].delta.content can be None (at the finish line)
                        response.choices[0].delta.content or ""
                        if response.choices
                        else ""
                    )
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

                st.session_state.with_search_engine_chat_messages.append(
                    {
                        "role": "assistant",
                        "content": full_response,
                        "metadata": {
                            "finish_reason": response.choices[0].finish_reason
                        },
                    }
                )
                if show_metadata:
                    st.caption(
                        f"Finish reason: {st.session_state.with_search_engine_chat_messages[-1]['metadata']['finish_reason']}"
                    )

            except openai.BadRequestError as e:
                (
                    error_message,
                    error_reason,
                ) = utils.extract_error_from_openai_BadRequestError(e)

                message_placeholder.error(error_message)

                st.session_state.with_search_engine_chat_messages.append(
                    {
                        "role": "error",
                        "content": error_message,
                        "metadata": {"error_reason": error_reason},
                    }
                )
                if show_metadata:
                    st.caption(f"Error reason: {error_reason}")

# TODO: maybe summarize content for the file name
download_button.download_button(
    "Download current chat history",
    json.dumps(
        st.session_state.with_search_engine_chat_messages, indent=4, ensure_ascii=False
    ),
    f"history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
    mime="text/plain",
    disabled=not st.session_state.with_search_engine_chat_messages,
)
