import streamlit as st
import tiktoken
import utils

st.set_page_config(
    page_title="Tokenization",
)


"""
# Tokenization

- [Tokenizer | OpenAI Platform](https://platform.openai.com/tokenizer)
- [Tiktokenizer](https://tiktokenizer.vercel.app/)

- [What are tokens and how to count them? | OpenAI Help Center](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
- [Pricing](https://openai.com/pricing)
- [openai-cookbook/examples/How_to_count_tokens_with_tiktoken.ipynb at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)
"""

model_name = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
tokenizer = tiktoken.encoding_for_model(model_name)

with st.form("tokenizer"):
    text = st.text_area("Text")
    if submit := st.form_submit_button("Calculate"):
        token_integers = tokenizer.encode(text)
        num_token = len(token_integers)
        token_bytes = [
            tokenizer.decode_single_token_bytes(token) for token in token_integers
        ]

if submit:
    st.write("Total token:", num_token)
    st.write("Token integers:", token_integers)
    st.write("Token bytes:", token_bytes)
