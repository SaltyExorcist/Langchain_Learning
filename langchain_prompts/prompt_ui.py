from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

os.environ['HF_HOME'] = 'G:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

st.header("Research Tool")
user_input=st.text_input("Enter your prompt")

if st.button("summarize"):
    result=model.invoke(user_input)
    st.text(result.content)


