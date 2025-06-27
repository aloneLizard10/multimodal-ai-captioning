import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Set page config
st.set_page_config(page_title="Multimodal AI Poet", layout="centered")

st.title("🖼️📜 Multimodal AI: Image to Poetic Caption")
st.write("Upload an image and get a poetic description using AI (BLIP + GPT-2) — completely offline & free!")

# Load BLIP model and processor
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    return processor, model

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

# Generate base caption using BLIP
def generate_base_caption(image):
    processor, model = load_blip_model()
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Enhance caption with GPT-2 poetic style
def enhance_caption_with_gpt2(base_caption):
    tokenizer, model = load_gpt2()
    prompt = f"Write a short poetic line based on this: {base_caption}\nPoem:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    poetic_line = generated.replace(prompt, "").strip()
    return poetic_line

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating base caption..."):
        base_caption = generate_base_caption(image)
        st.write("**Base Caption:**", base_caption)

    with st.spinner("Enhancing to poetic style..."):
        poetic_caption = enhance_caption_with_gpt2(base_caption)
        if poetic_caption:
            st.markdown("**🌸 Poetic Caption:**")
            st.success(poetic_caption)
        else:
            st.error("Poetic caption not generated.")
