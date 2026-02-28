import streamlit as st
import torch
import re
import os
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import PyPDF2
from rephrase_with_gemini import BloomsGeminiRephraser

# Hardcoded API key provided by user for demonstration
GEMINI_API_KEY = "AIzaSyCExJ4LKcB7JKkcnpHzgiRKU0MPn3vMtgU"

# ─── Page Config ───
st.set_page_config(
    page_title="Bloom's Taxonomy QGen Comparison",
    page_icon="🎓",
    layout="wide",
)

# ─── Custom Styles ───
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #1f2937; }
    .main { padding: 2rem; }
    .stSlider { padding-bottom: 2rem; }
    h1, h2, h3, h4, p, label { color: #111827 !important; }
    .st-emotion-cache-16idsys p { font-size: 1.1rem; color: #1f2937; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #e5e7eb; }
    .stSidebar [data-testid="stMarkdownContainer"] p { color: #374151 !important; }
    .bloom-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
        color: #ffffff !important;
        margin-right: 0.5rem;
    }
    .tag-remember { background-color: #4A90E2; }
    .tag-understand { background-color: #50E3C2; }
    .tag-apply { background-color: #F5A623; }
    .tag-analyze { background-color: #D0021B; }
    .tag-evaluate { background-color: #BD10E0; }
    .tag-create { background-color: #9013FE; }
</style>
""", unsafe_allow_html=True)

# ─── Model Configs ───
MODEL_CONFIGS = {
    "Option A: 6-Level (Augmented)": {
        "path": "/home/harsh/projects/majorProject/question_gen_blooms_6level_model",
        "type": "6level",
        "description": "Balanced across all 6 original levels (Remember, Understand, Apply, Analyze, Evaluate, Create). Used data augmentation to fix underrepresented levels.",
        "levels": {
            1: "Remember",
            2: "Understand",
            3: "Apply",
            4: "Analyze",
            5: "Evaluate",
            6: "Create"
        },
        "level_descs": {
            1: "Recall facts and basic concepts",
            2: "Explain ideas or concepts",
            3: "Use information in new situations",
            4: "Draw connections among ideas",
            5: "Justify a stand or decision",
            6: "Produce new or original work"
        }
    },
    "Option B: 4-Level (Consolidated)": {
        "path": "/home/harsh/projects/majorProject/question_gen_blooms_4level_model",
        "type": "4level",
        "description": "Merged similar levels (L1+L2, L5+L6) to create 4 balanced categories. Higher reliability in level distinction.",
        "levels": {
            1: "Remember & Understand",
            2: "Apply",
            3: "Analyze",
            4: "Evaluate & Create"
        },
        "level_descs": {
            1: "Recall and comprehension of concepts",
            2: "Solution provided through application",
            3: "Deep analysis and critical pattern finding",
            4: "Critical judgment and creative synthesis"
        }
    }
}

# ─── Helper Functions ───

@st.cache_resource
def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small",
        torch_dtype=torch.float32, # Standardized on FP32 for stability
        device_map=None
    ).to(device)
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    return model, tokenizer, device

def init_gemini_rephraser(api_key):
    # No @st.cache_resource here so it forces a fresh load of the new google-genai logic
    try:
        from rephrase_with_gemini import BloomsGeminiRephraser
        return BloomsGeminiRephraser(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to load Gemini Rephraser: {e}")
        return None

def extract_text_from_pdf(pdf_file, start_page=0, skip_last=0):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    total_pages = len(pdf_reader.pages)
    
    end_page = total_pages - skip_last
    # Safety bounds
    start_page = max(0, min(start_page, total_pages - 1))
    end_page = max(start_page + 1, min(end_page, total_pages))
    
    for i in range(start_page, end_page):
        text += pdf_reader.pages[i].extract_text() + "\n"
    return text

def chunk_text(text, max_chars=800):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current, current_len = [], 0
    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            current, current_len = [sent], len(sent)
        else:
            current.append(sent)
            current_len += len(sent)
    if current: chunks.append(" ".join(current))
    return chunks

def generate_question(model, tokenizer, device, text, level_name):
    prompt = f"generate a {level_name.lower()} level question: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ─── Sidebar: Model Selection ───
st.sidebar.title("🛠 Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose Bloom's Model",
    options=list(MODEL_CONFIGS.keys())
)
config = MODEL_CONFIGS[selected_model_name]

st.sidebar.info(config["description"])

# ─── Main UI ───
st.title("🎓 Bloom's Taxonomy QGen")
st.subheader("Redesigned Question Generator")

with st.expander("📚 How to use", expanded=False):
    st.write("""
    1. **Upload a PDF** or paste text from a textbook.
    2. Use the **Bloom's Level Slider** to select the target cognitive complexity.
    3. Click **Generate** to create a question conditioned on that level.
    """)

# Input Method
tab_upload, tab_text = st.tabs(["📄 Upload PDF", "✍️ Paste Text"])

source_text = ""
with tab_upload:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        start_page = st.number_input("Start Page (0-indexed)", min_value=0, value=0)
    with col_opt2:
        skip_last = st.number_input("Pages to Skip at End", min_value=0, value=0)
        
    if uploaded_file:
        source_text = extract_text_from_pdf(uploaded_file, start_page=start_page, skip_last=skip_last)
        st.success(f"Extracted {len(source_text)} characters from PDF")

with tab_text:
    source_text = st.text_area("Paste content here", height=200, value=source_text if source_text else "")

# Level Selection
if source_text:
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### 🎚 Bloom's Level")
        max_level = len(config["levels"])
        selected_level_idx = st.select_slider(
            "Target Complexity",
            options=range(1, max_level + 1),
            value=1,
            format_func=lambda x: config["levels"][x]
        )
        level_name = config["levels"][selected_level_idx]
        st.info(f"**{level_name}**: {config['level_descs'][selected_level_idx]}")

    with col2:
        st.write("### ⚙️ Generation Options")
        num_questions = st.number_input("Questions to generate", min_value=1, max_value=5, value=1)
        if st.button("🚀 Generate Questions", type="primary"):
            if not source_text.strip():
                st.error("Please provide some text first!")
            else:
                with st.spinner("Loading AI model..."):
                    model, tokenizer, device = load_model(config["path"])
                    rephraser = init_gemini_rephraser(GEMINI_API_KEY)
                
                chunks = chunk_text(source_text)
                # Randomly pick chunks to generate from
                sampled_chunks = random.sample(chunks, min(len(chunks), num_questions))
                
                st.write("### ✨ Results")
                for i, chunk in enumerate(sampled_chunks):
                    with st.status(f"Generating question {i+1}...", expanded=True):
                        # Show chunk preview
                        st.caption(f"Source: {chunk[:100]}...")
                        raw_question = generate_question(model, tokenizer, device, chunk, level_name)
                        
                        # Use Gemini to rephrase if available
                        if rephraser:
                            st.caption("_Refining language with Gemini..._")
                            question = rephraser.rephrase(chunk, raw_question)
                        else:
                            question = raw_question
                        
                        # Extract tag for styling if present
                        tag_match = re.match(r'^\[(.*?)\]', question)
                        clean_q = question
                        tag_style = "tag-remember" # Default
                        
                        if tag_match:
                            tag_text = tag_match.group(1).lower()
                            clean_q = question[tag_match.end():].strip()
                            if "remember" in tag_text: tag_style = "tag-remember"
                            elif "understand" in tag_text: tag_style = "tag-understand"
                            elif "apply" in tag_text: tag_style = "tag-apply"
                            elif "analyze" in tag_text: tag_style = "tag-analyze"
                            elif "evaluate" in tag_text: tag_style = "tag-evaluate"
                            elif "create" in tag_text: tag_style = "tag-create"
                            
                            st.markdown(f"<span class='bloom-tag {tag_style}'>{tag_match.group(1)}</span>", unsafe_allow_html=True)
                        
                        st.markdown(f"#### {clean_q}")
                        
else:
    st.info("👈 Start by uploading a PDF or pasting text in the tabs above.")

# Footer
st.sidebar.divider()
st.sidebar.caption("Redesign Iteration 2 • Built with T5-Small + LoRA")
st.sidebar.markdown(f"**Mode**: {config['type'].upper()}")
