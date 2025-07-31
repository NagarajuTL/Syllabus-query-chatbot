import streamlit as st
import os
import json
import asyncio
import nest_asyncio
import tempfile
import boto3
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

import os

# Load environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION = os.getenv("REGION")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Allow Streamlit + asyncio to work
nest_asyncio.apply()
asyncio.set_event_loop(asyncio.new_event_loop())



s3 = boto3.client("s3",
    region_name=REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

META_FILE = "metadata.json"

#extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Save FAISS index to S3
def save_and_upload_faiss_to_s3(text_chunks, branch, year):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store.save_local(tmpdir)  

        s3_prefix = f"faiss_index/{branch}_{year}"
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, tmpdir)
                s3_path = os.path.join(s3_prefix, relative_path)
                s3.upload_file(local_path, BUCKET_NAME, s3_path)

# Update metadata file
def update_metadata(branch, year):
    if not os.path.exists(META_FILE):
        metadata = {}
    else:
        with open(META_FILE, "r") as f:
            metadata = json.load(f)

    if branch in metadata:
        if year not in metadata[branch]:
            metadata[branch].append(year)
    else:
        metadata[branch] = [year]

    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

# Upload metadata file to S3
def upload_metadata_to_s3():
    s3.upload_file(META_FILE, BUCKET_NAME, META_FILE)


#Streamlit UI
st.title("📥 Admin Panel - Upload Syllabus PDF")

branch = st.selectbox("Select Branch", ["CSE", "EEE", "ECE", "MECH"])
year = st.selectbox("Select Year", ["2022-23", "2023-24", "2024-25"])
pdf = st.file_uploader("Upload Syllabus PDF", type="pdf")

if st.button("Submit & Process") and pdf:
    with st.spinner("Processing..."):
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text)

        save_and_upload_faiss_to_s3(chunks, branch, year)
        update_metadata(branch, year)
        upload_metadata_to_s3()

        st.success(f"Syllabus for {branch} - {year} uploaded  successfully! ")
