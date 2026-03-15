import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------------------
# Load FLAN-T5-LARGE
# -------------------------------
model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# -------------------------------
# Process PDF
# -------------------------------
def process_pdf(pdf_file):

    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    cleaned_texts = []

    for doc in documents:
        text = doc.page_content

        # Remove Question section
        if "Question" in text:
            text = text.split("Question")[0]

        # Remove Answer section
        if "Answer:" in text:
            text = text.split("Answer:")[0]

        cleaned_texts.append(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    docs = text_splitter.create_documents(cleaned_texts)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    return db


# -------------------------------
# Chat Function
# -------------------------------
def chat(pdf, question):

    if pdf is None:
        return "Please upload a PDF first."

    if not question.strip():
        return "Please enter a question."

    db = process_pdf(pdf)

    docs = db.similarity_search(question, k=4)

    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful AI tutor.

Read the information and explain the answer in your own words.

Rules:
- Do NOT copy sentences directly from the text.
- Understand the meaning first.
- Explain clearly in 2-3 sentences.
- Rewrite the answer in simple words.

Information:
{context}

Question:
{question}

Explain the answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=120,
    temperature=0.6,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.4,
    no_repeat_ngram_size=3
)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]

    return answer.strip()


# -------------------------------
# Gradio Interface
# -------------------------------
interface = gr.Interface(
    fn=chat,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Ask Question")
    ],
    outputs="text",
    title="Advanced PDF Chatbot (FLAN-T5-Large + RAG)"
)

interface.launch(share=True)