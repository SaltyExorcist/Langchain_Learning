from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get token from environment
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Check if token loaded
if not hf_token:
    raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN not found in environment variables. Please check your .env file.")

# Optional: Set Hugging Face cache directory
os.environ['HF_HOME'] = 'G:/huggingface_cache'

# Load model with explicit token
llm = HuggingFacePipeline.from_model_id(
    model_id='google/gemma-2-2b-it',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    ),
    huggingfacehub_api_token=hf_token  # Explicitly pass token
)

# Wrap with LangChain
model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

# Run the chain
result = chain.invoke({'topic': 'black hole'})
print(result)
