import subprocess
from dotenv import load_dotenv
import os
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

# Make sure to load the environment variables
load_dotenv()

file_path = r'C:\\Users\\ihamz\\curestock\\A-sample-prescription-image-in-grayscale-version.png'
api_key = os.getenv('TOGETHER_API_KEY')

# Add error handling for missing API key
if api_key is None:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")

result = subprocess.run(
    ['node', 'ocrScript.js', file_path, api_key],
    capture_output=True,
    text=True
)

print(result.stdout)

def process_and_query(text, query):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        length_function=len,
    )
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = docsearch.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return result

query = """Extract only the medication-related information from the prescription text in a structured format. For each medication, include:

- Brand name (in parentheses)
- Generic name
- Strength
- Dosage form
- Quantity (number after #)
- Instructions (after Sig:)

Format the output as a JSON object with the following structure:
{
    "medications": [
        {
            "brand_name": "",
            "generic_name": "",
            "strength": "",
            "dosage_form": "",
            "quantity": "",
            "instructions": ""
        }
    ]
}

Ignore all other information like patient details, physician details, license numbers, etc. Remove any duplicates in the medications list. Only include unique medication entries."""

text = result.stdout

extracted_result = process_and_query(text, query)
print(extracted_result)