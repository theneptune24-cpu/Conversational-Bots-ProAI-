<<<<<<< HEAD
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
    repetition_penalty=1.4W,
    no_repeat_ngram_size=3
)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]
=======
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import gradio as gr
from PIL import Image

model_id = "Qwen/Qwen2-VL-2B-Instruct"

print("Loading model... please wait")

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)

model.eval()

print("Model loaded successfully!")

def predict(image, question):

    if image is None:
        return "Please upload an image."

    # Fix image format
    image = image.convert("RGB")

    # Resize for faster processing
    image = image.resize((384, 384))

    # Create chat message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ],
        }
    ]

    # Convert to model input format
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    # Move inputs to CPU
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300
        )

    # Decode only the generated answer
    answer = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )[0]
>>>>>>> 96171d1c2df2032ed096d21e7ab26c9f44252b05

    return answer.strip()


<<<<<<< HEAD
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
=======
# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Ask a question about the image")
    ],
    outputs="text",
    title="Image Question Answering Chatbot",
    description="Upload an image and ask questions about it."
)

interface.launch()
>>>>>>> 96171d1c2df2032ed096d21e7ab26c9f44252b05
