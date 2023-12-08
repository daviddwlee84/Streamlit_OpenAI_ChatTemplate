import streamlit as st
from dotenv import load_dotenv
import os
import openai
import datetime
import json
import utils

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, "../.env"))

st.set_page_config(
    page_title="Simplest Chat (OpenAI)",
)

st.title("Simplest Chat (OpenAI)")


with st.expander("Settings"):
    do_streaming = st.checkbox("Streaming output", True)
    show_metadata = st.checkbox("Show metadata", True)
    initial_system_message = st.text_area(
        "System Message (modify this will clear history)", "How can I help you?"
    )

with st.sidebar:
    openai_selection = utils.generate_api_and_language_model_selection()


if (
    "simplest_chat_messages" not in st.session_state
    or st.session_state.simplest_chat_messages[0]['content'] != initial_system_message
):
    st.session_state["simplest_chat_messages"] = [
        # TODO: able to set system initial prompt (If reset prompt then clear history)
        {"role": "assistant", "content": initial_system_message, "metadata": {}}
    ]


download_button = st.empty()

# Render history messages
for msg in st.session_state.simplest_chat_messages:
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

    st.session_state.simplest_chat_messages.append(
        {"role": "user", "content": prompt, "metadata": {}}
    )
    st.chat_message("user").write(prompt)

    if not do_streaming:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = client.chat.completions.create(
                    model=st.session_state.model,
                    # Chat history
                    # TODO: don't send user message cause error
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.simplest_chat_messages
                        if m["role"] in {"user", "assistant"}
                    ],
                    temperature=st.session_state.temperature,
                    stream=False,
                )
                msg = response.choices[0].message
                st.session_state.simplest_chat_messages.append(
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

                st.session_state.simplest_chat_messages.append(
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
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.simplest_chat_messages
                        if m["role"] in {"user", "assistant"}
                    ],
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

                st.session_state.simplest_chat_messages.append(
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
                        f"Finish reason: {st.session_state.simplest_chat_messages[-1]['metadata']['finish_reason']}"
                    )

            except openai.BadRequestError as e:
                (
                    error_message,
                    error_reason,
                ) = utils.extract_error_from_openai_BadRequestError(e)

                message_placeholder.error(error_message)

                st.session_state.simplest_chat_messages.append(
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
    json.dumps(st.session_state.simplest_chat_messages, indent=4, ensure_ascii=False),
    f"history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
    mime="text/plain",
    disabled=not st.session_state.simplest_chat_messages,
)
