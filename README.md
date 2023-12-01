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
- [ ] Clear chat history for fresh new round
- [X] Error handling

Demonstration

- [ ] Chat with memory (different memory methods)
- [ ] Chain-of-thought and intermediate result visualization
- [ ] Agents selection

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

### OpenAI

- [openai-cookbook/examples/How_to_stream_completions.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb)

- Error response format
  - [response_format error · Issue #887 · openai/openai-python](https://github.com/openai/openai-python/issues/887)
