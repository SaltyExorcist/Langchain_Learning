from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

llm = HuggingFacePipeline.from_model_id(
    model_id='google/gemma-2-2b-it',
    task='text-generation',
    device=0,
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=500
    ) 
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()