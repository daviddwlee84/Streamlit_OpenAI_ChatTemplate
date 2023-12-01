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
