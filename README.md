# Streamlit OpenAI ChatTemplate

Simple chat App template using Streamlit Chat component with OpenAI Key. Which is useful as a starter template to build a chat app.

## Getting Started

```bash
pip install -r requirements.txt
```

```bash
# Optional
cp example.env .env
# Fill-in your keys
# ...
```

```bash
streamlit run Overview.py
```

## Todo

Functionality

- [X] Streaming user experience
  - [X] Option of streaming or not
- [ ] Option of print cost estimation (and other metadata)
- [ ] Language model parameters
  - [X] Model name
  - [X] Temperature
  - [ ] TopN
  - [ ] ...
- [ ] Download or preserve/resume chat history
  - [X] Download history: [st.download_button - Streamlit Docs](https://docs.streamlit.io/library/api-reference/widgets/st.download_button)
- [ ] Clear chat history for fresh new round
- [X] Error handling

Demonstration

- [ ] Basic usage
  - [ ] Able to set system prompt instead of "How can I help you?"
    - [LouisShark/chatgpt_system_prompt: store all agent's system prompt](https://github.com/LouisShark/chatgpt_system_prompt)
    - [linexjlin/GPTs: leaked prompts of GPTs](https://github.com/linexjlin/GPTs)
- [ ] Chat with memory (different memory methods)
- [ ] Chain-of-thought and intermediate result visualization
- [ ] Agents selection
- [ ] Retrieval Augmentation
  - [LlamaIndex - Data Framework for LLM Applications](https://www.llamaindex.ai/)
  - [run-llama/llama_index: LlamaIndex (formerly GPT Index) is a data framework for your LLM applications](https://github.com/run-llama/llama_index)

Integration

- [ ] Try moving from Streamlit to [Chainlit](https://chainlit.io/)
  - [Overview - Chainlit](https://docs.chainlit.io/get-started/overview)
  - [Chainlit/chainlit: Build Python LLM apps in minutes ⚡️](https://github.com/Chainlit/chainlit)

## Resources

### Streamlit

- [Build a basic LLM chat app - Streamlit Docs](https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps)
- [Chat elements - Streamlit Docs](https://docs.streamlit.io/library/api-reference/chat)

LangChain x Streamlit

- [Streamlit | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/callbacks/streamlit)
- [Streamlit • Generative AI](https://streamlit.io/generative-ai)
  - [streamlit/llm-examples: Streamlit LLM app examples for getting started](https://github.com/streamlit/llm-examples/)

Deploy to Streamlit Community Cloud

- [Deploy your app - Streamlit Docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- [App dependencies - Streamlit Docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Configuration - Streamlit Docs](https://docs.streamlit.io/library/advanced-features/configuration)
- [Secrets management - Streamlit Docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)
  - Copy `.env` settings to the Streamlit App Settings > Secrets

### OpenAI & LangChain

- [openai/openai-python: The official Python library for the OpenAI API](https://github.com/openai/openai-python)
- [Chat models | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/model_io/chat/)
- Streaming
  - [openai-cookbook/examples/How_to_stream_completions.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)
  - [Langchain/Openai Streaming 101 in Python | by Esther is a confused human being | LLM Projects & Philosophy on How to Build Fast | Medium](https://medium.com/llm-projects/langchain-openai-streaming-101-in-python-edd60e84c9ca)
    - Concept: Event-Driven API
    - [Custom Response - HTML, Stream, File, others - FastAPI](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
    - [**Streaming | 🦜️🔗 Langchain**](https://python.langchain.com/docs/modules/model_io/chat/streaming)
      - [Chat models (streaming availability) | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/chat/)
    - [**examples/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb at master · pinecone-io/examples**](https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/09-langchain-streaming/09-langchain-streaming.ipynb)
    - [ajndkr/lanarky: The web framework for building LLM microservices](https://github.com/ajndkr/lanarky) (deprecated: [fastapi-async-langchain · PyPI](https://pypi.org/project/fastapi-async-langchain/))
- Error response format
  - [response_format error · Issue #887 · openai/openai-python](https://github.com/openai/openai-python/issues/887)

### Others

- [BerriAI/litellm: Call all LLM APIs using the OpenAI format. Use Bedrock, Azure, OpenAI, Cohere, Anthropic, Ollama, Sagemaker, HuggingFace, Replicate (100+ LLMs)](https://github.com/BerriAI/litellm)
  - [LiteLLM - Getting Started | liteLLM](https://docs.litellm.ai/docs/)
  - [Providers | liteLLM](https://docs.litellm.ai/docs/providers/)
- [KillianLucas/open-interpreter: A natural language interface for computers](https://github.com/KillianLucas/open-interpreter?tab=readme-ov-file )
