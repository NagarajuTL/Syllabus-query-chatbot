# Syllabus Query Chatbot

This is a chatbot project where students can ask questions related to their college syllabus.  
Admins can upload syllabus PDFs for different branches and years. Students just pick their branch and year and start asking!


# What It Does

- Admin uploads syllabus PDFs (like EEE 2022, CSE 2023, etc.)
- The bot reads and stores the syllabus smartly using embeddings (Google Gemini) and FAISS
- Students select their branch & year and ask questions
- The chatbot finds the most relevant answer based on the syllabus content


# Tools & Tech Used

- **Streamlit** – for the admin and student UI
- **LangChain** – handles how the bot processes and answers questions
- **Google Gemini API** – generates embeddings and answers
- **FAISS** – stores the syllabus in vector form for fast searching
- **AWS S3** – used to store and load syllabus indexes online


