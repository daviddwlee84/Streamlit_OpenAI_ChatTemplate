from typing import Annotated, Literal
import streamlit as st
from dotenv import load_dotenv
import os
import datetime
import json
import utils
from langchain_openai import ChatOpenAI, AzureChatOpenAI

from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver

# ImportError: cannot import name 'ReadOnly' from 'typing_extensions' (C:\Python311\Lib\site-packages\typing_extensions.py)
# pip install --upgrade typing-extensions
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# https://langchain-ai.github.io/langgraph/tutorials/introduction/

curr_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(curr_dir, "../.env"))

st.set_page_config(
    page_title="Simplest Agent (LangGraph)",
)


st.title("Simplest Agent (LangGraph)")

st.caption(
    "This example contains DuckDuckGo tool + Human-in-the-loop. For more LangGraph details check their [website](https://langchain-ai.github.io/langgraph/tutorials/introduction/)."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # This flag is new
    ask_human: bool


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


with st.sidebar:
    openai_selection = utils.generate_api_and_language_model_selection()


with st.expander("Settings"):
    preserve_all_history = st.checkbox(
        "Preserve all history",
        True,
        help='If unchecked, we will only keep track of "visible" part of the message.',
    )
    show_all_messages = st.checkbox(
        "Show all messages", False, help="Also show all intermediate tool calls and tool messages"
    )
    do_streaming = st.checkbox("Streaming output", False)
    initial_system_message = st.text_area(
        "System Message (modify this will clear history)", "How can I help you?"
    )

if (
    "langgraph_agents_chat_history" not in st.session_state
    or st.session_state.langgraph_agents_chat_history[0].content
    != initial_system_message
):
    st.session_state["langgraph_agents_chat_history"] = [
        SystemMessage(content=initial_system_message)
    ]


@st.fragment
def search_test():
    # https://python.langchain.com/docs/integrations/tools/ddg/
    # RateLimit error might caused by `duckduckgo-search` version
    # https://github.com/crewAIInc/crewAI/issues/136
    search = DuckDuckGoSearchResults()
    query = st.text_input("Query")
    if st.button("Search"):
        st.write(search.invoke(query))


with st.expander("Search Test"):
    search_test()


@st.cache_resource
def get_graph(openai_selection: str, cache_key: str = "") -> CompiledStateGraph:
    if openai_selection == "OpenAI":
        if not st.session_state.openai_api_key:
            st.warning("🥸 Please add your OpenAI API key to continue.")
            st.stop()

        llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            temperature=st.session_state.temperature,
            model=st.session_state.model,
        )

    elif openai_selection == "Azure OpenAI":
        if not st.session_state.azure_openai_api_key:
            st.warning("🥸 Please add your Azure OpenAI API key to continue.")
            st.stop()

        llm = AzureChatOpenAI(
            api_key=st.session_state.azure_openai_api_key,
            azure_endpoint=st.session_state.azure_openai_endpoint,
            azure_deployment=st.session_state.azure_openai_deployment_name,
            api_version=st.session_state.azure_openai_version,
            temperature=st.session_state.temperature,
            model=st.session_state.model,
        )

    tool = DuckDuckGoSearchResults()
    tools = [tool]
    # We can bind the llm to a tool definition, a pydantic model, or a json schema
    llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

    def chatbot(state: State) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        ask_human = False
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == RequestAssistance.__name__
        ):
            ask_human = True
        return {"messages": [response], "ask_human": ask_human}

    def create_response(response: str, ai_message: AIMessage) -> ToolMessage:
        return ToolMessage(
            content=response,
            tool_call_id=ai_message.tool_calls[0]["id"],
        )

    def human_node(state: State) -> dict:
        new_messages = []
        if not isinstance(state["messages"][-1], ToolMessage):
            # Typically, the user will have updated the state during the interrupt.
            # If they choose not to, we will include a placeholder ToolMessage to
            # let the LLM continue.
            new_messages.append(
                create_response("No response from human.", state["messages"][-1])
            )
        return {
            # Append the new messages
            "messages": new_messages,
            # Unset the flag
            "ask_human": False,
        }

    def select_next_node(state: State):
        if state["ask_human"]:
            return "human"
        # Otherwise, we can route as before
        return tools_condition(state)

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=[tool]))
    graph_builder.add_node("human", human_node)
    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("human", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["human"],
    )
    return graph


graph = get_graph(openai_selection, initial_system_message)
st.image(graph.get_graph().draw_mermaid_png())

download_button = st.empty()


# Render history messages
for msg in utils.convert_langchain_to_openai_message_history(
    st.session_state.langgraph_agents_chat_history, include_tool=show_all_messages
):
    role = msg["role"]
    if not show_all_messages and role.startswith("tool"):
        continue
    if msg["role"] in ["error", "tool", "tool_call"]:
        role = "assistant"
    with st.chat_message(role):
        if show_all_messages:
            if msg["role"] not in {"error", "user"}:
                st.text(msg["role"])
        if msg["role"] in {"user", "assistant", "tool"}:
            st.markdown(msg["content"])
        elif msg["role"] == "tool_call":
            st.json(msg["content"])
        else:
            st.error(msg["content"])

if prompt := st.chat_input():
    st.session_state.langgraph_agents_chat_history.append(HumanMessage(content=prompt))
    current_msg_length = len(st.session_state.langgraph_agents_chat_history)
    st.chat_message("user").write(prompt)

    if not do_streaming:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = graph.invoke(
                {"messages": st.session_state.langgraph_agents_chat_history},
                config={"configurable": {"thread_id": "1"}},
                stream_mode="values",
            )
            msg = response["messages"][-1]
            if preserve_all_history:
                st.session_state.langgraph_agents_chat_history = response["messages"]
            else:
                st.session_state.langgraph_agents_chat_history.append(
                    msg["messages"][-1]
                )
            if show_all_messages:
                for msg in response["messages"][current_msg_length:]:
                    st.text(msg.type)
                    if not msg.content:
                        # should be tool call
                        # st.json(msg.additional_kwargs)
                        st.json(msg.tool_calls)
                    else:
                        st.write(msg.content)
            else:
                message_placeholder.write(msg.content)
    else:
        # graph.stream()
        raise NotImplementedError()

# TODO: maybe summarize content for the file name
download_button.download_button(
    "Download current chat history",
    # TypeError: Object of type SystemMessage is not JSON serializable
    json.dumps(
        utils.convert_langchain_to_openai_message_history(
            st.session_state.langgraph_agents_chat_history,
        ),
        indent=4,
        ensure_ascii=False,
    ),
    f"history_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
    mime="text/plain",
    disabled=not st.session_state.langgraph_agents_chat_history,
)
