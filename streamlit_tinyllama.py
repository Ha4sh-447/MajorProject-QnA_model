import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import torch
import PyPDF2
from peft import PeftModel
import gc

# -----------------------------
# Load fine-tuned model + embeddings model
# -----------------------------
@st.cache_resource
def load_models():
    model_path = "./question_gen_tinyllama"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    
    # Load semantic model
    semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    base_model.eval()
    
    return tokenizer, base_model, semantic_model

tokenizer, model, semantic_model = load_models()

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_file, skip_last_pages=7):
    """
    Extract text from uploaded PDF file, skipping the last N pages.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        pages_to_read = total_pages - skip_last_pages
        
        if pages_to_read <= 0:
            st.error(f"PDF has only {total_pages} pages. Cannot skip {skip_last_pages} pages.")
            return None
        
        text = ""
        for page_num in range(pages_to_read):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# -----------------------------
# Text Chunking with Overlap
# -----------------------------
def chunk_text(text, max_words=400, overlap_words=50):
    """
    Split text into chunks of max_words with overlap_words between chunks.
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap_words
        
        if i + overlap_words >= len(words):
            break
    
    return chunks

# -----------------------------
# Question Generation
# -----------------------------
def generate_questions_from_chunk(chunk, model, tokenizer):
    """
    Generate questions from a single chunk of text using TinyLlama.
    """
    # Format prompt for TinyLlama
    prompt = f"<|im_start|>user\n{chunk}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract assistant's response
    if "<|im_start|>assistant" in generated_text:
        generated_text = generated_text.split("<|im_start|>assistant")[-1].strip()
    if "<|im_end|>" in generated_text:
        generated_text = generated_text.split("<|im_end|>")[0].strip()
    
    # Clean up if it starts with the user prompt
    if chunk[:50] in generated_text:
        generated_text = generated_text.replace(chunk, "").strip()
    
    return generated_text

# -----------------------------
# Semantic Relevancy
# -----------------------------
def calculate_relevance(chunk, question, semantic_model):
    """
    Calculate semantic similarity between chunk and generated question.
    """
    emb_chunk = semantic_model.encode(chunk, convert_to_tensor=True)
    emb_question = semantic_model.encode(question, convert_to_tensor=True)
    similarity = util.cos_sim(emb_chunk, emb_question)
    return float(similarity[0][0]) * 100

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="TinyLlama Question Generator", page_icon="🧠", layout="wide")

st.title("🧠 TinyLlama Question Generator from PDF")
st.markdown("Upload a PDF chapter and generate semantically relevant questions using a fine-tuned TinyLlama 1.1B model.")

st.divider()

# -----------------------------
# Class fixed
# -----------------------------
st.subheader("Class: 11 (Fixed)")

# -----------------------------
# Subject dropdown
# -----------------------------
subject = st.selectbox("Select Subject", ["", "Biology", "Chemistry", "Physics"])

# -----------------------------
# Chapter mapping (clean titles)
# -----------------------------
def clean_name(filename: str) -> str:
    """Convert snake_case filenames to Title Case (without .pdf)."""
    name = filename.replace(".pdf", "").replace("_", " ").strip()
    return name.title()

chapters = {
    "Biology": [
        "anatomy_of_flowering_plants.pdf",
        "animal_kingdom.pdf",
        "biological_classification.pdf",
        "biomolecules.pdf",
        "body_fluids_and_circulation.pdf",
        "breathing_and_exchange_of_gases.pdf",
        "cell.pdf",
        "cell_cycle_and_cell_division.pdf",
        "chemincal_coordination_and_integration.pdf",
        "excetory_products_and_their_coodination.pdf",
        "locomotion_and_movement.pdf",
        "morphology_of_flowering_plants.pdf",
        "neural_control_and_elimination.pdf",
        "photosynthesis_in_higher_plants.pdf",
        "plant_growth_and_development.pdf",
        "plant_kingdom.pdf",
        "respiration_in_plants.pdf",
        "structural_organization_in_animals.pdf",
        "the_living_world.pdf",
    ],
    "Chemistry": [
        "chemical_bonding_molecular_structure.pdf",
        "classification_of_elements_periodicity.pdf",
        "equilibrium.pdf",
        "hydrocarbons.pdf",
        "organic_chem_some_basic_principals.pdf",
        "redox_reactions.pdf",
        "some_basic_concepts.pdf",
        "structure_of_atom.pdf",
        "thermodynamics.pdf",
    ],
    "Physics": [
        "gravitation.pdf",
        "kinetic_theory.pdf",
        "laws_motion.pdf",
        "motion_in_plane.pdf",
        "motion_straight_line.pdf",
        "oscillations.pdf",
        "properties_of_fluids.pdf",
        "properties_of_solids.pdf",
        "rotational_motion.pdf",
        "thermal_properties.pdf",
        "thermodynamics.pdf",
        "units_measurements.pdf",
        "waves.pdf",
        "work_energy_power.pdf",
    ],
}

# Convert chapter names to Title Case (without .pdf)
for subj in chapters:
    chapters[subj] = [clean_name(ch) for ch in chapters[subj]]

chapter = None
if subject:
    chapter = st.selectbox("Select Chapter", [""] + chapters[subject])

# -----------------------------
# PDF Upload
# -----------------------------
st.subheader("📄 Upload PDF Chapter")
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

# Configuration
col1, col2 = st.columns(2)
with col1:
    max_chunk_words = st.number_input("Max words per chunk", min_value=100, max_value=500, value=400, step=50)
with col2:
    overlap_words = st.number_input("Overlap words between chunks", min_value=0, max_value=100, value=50, step=10)

skip_last_pages = st.number_input("Skip last N pages", min_value=0, max_value=20, value=7, step=1)

# -----------------------------
# Generate button
# -----------------------------
if st.button("🔮 Generate Questions from PDF", type="primary"):
    if not subject or not chapter:
        st.error("Please select both subject and chapter.")
    elif not uploaded_file:
        st.error("Please upload a PDF file.")
    else:
        # Extract text from PDF
        with st.spinner("📖 Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file, skip_last_pages=skip_last_pages)
        
        if pdf_text:
            # Count words
            total_words = len(pdf_text.split())
            st.success(f"✅ Extracted {total_words} words from PDF (skipped last {skip_last_pages} pages)")
            
            # Chunk the text
            with st.spinner("✂️ Chunking text..."):
                chunks = chunk_text(pdf_text, max_words=max_chunk_words, overlap_words=overlap_words)
            
            st.info(f"📊 Created {len(chunks)} chunks for processing")
            
            # Create containers for displaying results
            st.divider()
            st.subheader("📝 Generated Questions")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Store all questions
            all_questions = []
            
            # Process each chunk
            for idx, chunk in enumerate(chunks, 1):
                # Update progress
                progress_bar.progress(idx / len(chunks))
                status_text.text(f"Processing chunk {idx}/{len(chunks)}...")
                
                # Create expander for this chunk
                with st.expander(f"🔹 Chunk {idx} ({len(chunk.split())} words)", expanded=(idx == 1)):
                    # Show chunk preview
                    st.caption("**Chunk Text Preview:**")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                    
                    # Generate questions
                    with st.spinner(f"Generating questions for chunk {idx}..."):
                        questions = generate_questions_from_chunk(chunk, model, tokenizer)
                    
                    # Calculate relevance
                    relevance = calculate_relevance(chunk, questions, semantic_model)
                    
                    # Display questions
                    st.markdown("**Generated Questions:**")
                    st.markdown(f"> {questions}")
                    
                    # Show relevance score
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.progress(relevance / 100)
                    with col2:
                        st.metric("Relevance", f"{relevance:.1f}%")
                    
                    # Store questions
                    all_questions.append({
                        'chunk_id': idx,
                        'questions': questions,
                        'relevance': relevance
                    })
                
                # Clear CUDA cache periodically
                if idx % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # Clear progress
            progress_bar.progress(100)
            status_text.text("✅ All chunks processed!")
            
            # Summary section
            st.divider()
            st.subheader("📊 Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", len(chunks))
            with col2:
                st.metric("Total Words", total_words)
            with col3:
                avg_relevance = sum(q['relevance'] for q in all_questions) / len(all_questions)
                st.metric("Avg Relevance", f"{avg_relevance:.1f}%")
            
            # Download all questions
            st.divider()
            st.subheader("💾 Download All Questions")
            
            # Format all questions for download
            download_text = f"Generated Questions for {subject} - {chapter}\n"
            download_text += "=" * 60 + "\n\n"
            
            for q_data in all_questions:
                download_text += f"Chunk {q_data['chunk_id']} (Relevance: {q_data['relevance']:.1f}%)\n"
                download_text += "-" * 60 + "\n"
                download_text += f"{q_data['questions']}\n\n"
            
            st.download_button(
                label="📥 Download All Questions as TXT",
                data=download_text,
                file_name=f"{subject}_{chapter}_questions.txt",
                mime="text/plain"
            )

st.markdown("---")
st.caption("Powered by TinyLlama 1.1B + SentenceTransformer — © 2025 Harsh Saindane")
