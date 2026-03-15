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

    return answer.strip()


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