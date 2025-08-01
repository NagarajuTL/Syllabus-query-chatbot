import streamlit as st
import os
import json
import asyncio
import nest_asyncio
import boto3

from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

import os

nest_asyncio.apply()
asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("REGION")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
META_FILE = "metadata.json"
INDEX_FOLDER = "faiss_index"

s3 = boto3.client("s3",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)


def download_metadata_from_s3():
    s3.download_file(BUCKET_NAME, META_FILE, META_FILE)

# Download FAISS index from S3
def download_faiss_index(branch, year):
    prefix = f"{INDEX_FOLDER}/{branch}_{year}"
    local_folder = f"{INDEX_FOLDER}/{branch}_{year}"
    os.makedirs(local_folder, exist_ok=True)

    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    for obj in response.get("Contents", []):
        s3_key = obj["Key"]
        local_path = s3_key
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET_NAME, s3_key, local_path)

#load metadata 
def load_available_data():
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, "r") as f:
        return json.load(f)

# Get vector store for a specific branch and year
def get_vector_store(branch, year):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = f"{INDEX_FOLDER}/{branch}_{year}"
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Get QA chain for answering questions
def get_qa_chain():
    prompt_template = """
    Answer the question using the context provided below. 
    If the answer is not in the context, reply with "Sorry not Available"
    Don't mention the number of hours.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Answer a question using the vector store and QA chain
def answer_question(branch, year, user_question):
    download_faiss_index(branch, year)
    db = get_vector_store(branch, year)
    docs = db.similarity_search(user_question)
    chain = get_qa_chain()
    result = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return result["output_text"]

# Streamlit UI
st.set_page_config(page_title="College Syllabus Chatbot")
st.title("🎓 College Syllabus Query Bot")

download_metadata_from_s3()
metadata = load_available_data()
branches = list(metadata.keys())



branch = st.sidebar.selectbox("Select Branch", branches)
year = st.sidebar.selectbox("Select Year", metadata.get(branch, []))
user_question = st.text_input("Ask a question about the syllabus")

if st.button("Get Answer") and user_question:
    with st.spinner("Finding answer..."):
        try:
            response = answer_question(branch, year, user_question)
            st.success("Answer:")
            st.write(response)
        except Exception as e:
            st.error("Something went wrong while answering. Check logs.")
            st.exception(e)
