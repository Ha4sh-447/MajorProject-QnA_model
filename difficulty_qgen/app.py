#!/usr/bin/env python3
"""
Streamlit app for difficulty-conditioned question generation from PDFs.
Uses the fine-tuned LoRA model with creative sampling for novel output.
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
import PyPDF2

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# ─── Constants ───────────────────────────────────────────────────────────────

DIFFICULTY_LABELS = {
    1: "Level 1 — Multiple Choice (MCQ)",
    2: "Level 2 — Very Short Answer",
    3: "Level 3 — Short Answer",
    4: "Level 4 — Long Answer",
}

MODEL_DIR = "../question_gen_difficulty_model"
BASE_MODEL = "google/flan-t5-small"
FALLBACK_DIR = "../question_gen_model_final"

# ─── Model Loading ───────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load the question generation model and the semantic similarity model."""
    try:
        if not HAS_PEFT:
            raise ImportError("peft not installed")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
        model.eval()
        model_type = "difficulty"
    except Exception:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(FALLBACK_DIR)
        model = T5ForConditionalGeneration.from_pretrained(FALLBACK_DIR)
        model_type = "basic"

    semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model, semantic_model, model_type


tokenizer, model, semantic_model, model_type = load_models()

# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file, skip_last_pages=7):
    """Extract text from PDF, skipping last N pages."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        total = len(reader.pages)
        pages = total - skip_last_pages
        if pages <= 0:
            st.error(f"PDF has only {total} pages, can't skip {skip_last_pages}")
            return None
        text = ""
        for i in range(pages):
            text += reader.pages[i].extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


def chunk_text(text, max_words=400, overlap_words=50):
    """Split text into overlapping chunks."""
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


def generate_questions(chunk, difficulty_level=None):
    """
    Generate questions from text chunk.
    Uses creative sampling for novel output when difficulty model is available.
    """
    if difficulty_level and model_type == "difficulty":
        prompt = f"generate level {difficulty_level} question: {chunk}"
    else:
        prompt = f"generate questions: {chunk}"

    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # Use sampling for creative, non-memorized output
    output_ids = model.generate(
        inputs,
        max_length=256,
        min_length=20,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=50,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def calculate_relevance(chunk, question):
    """Semantic similarity between chunk and generated question (0-100%)."""
    emb_c = semantic_model.encode(chunk, convert_to_tensor=True)
    emb_q = semantic_model.encode(question, convert_to_tensor=True)
    return float(util.cos_sim(emb_c, emb_q)[0][0]) * 100


def clean_name(filename):
    return filename.replace(".pdf", "").replace("_", " ").strip().title()


# ─── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Question Generator", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Question Generator from PDF")
st.markdown("Upload a PDF chapter and generate questions at your chosen difficulty level.")

if model_type == "difficulty":
    st.success("✅ Difficulty-conditioned model loaded — creative generation enabled")
else:
    st.warning("⚠️ Using fallback model — difficulty selection unavailable")

st.divider()

# Subject / Chapter selection
st.subheader("Class: 11 (Fixed)")
subject = st.selectbox("Select Subject", ["", "Biology", "Chemistry", "Physics"])

chapters = {
    "Biology": [
        "anatomy_of_flowering_plants.pdf", "animal_kingdom.pdf",
        "biological_classification.pdf", "biomolecules.pdf",
        "body_fluids_and_circulation.pdf", "breathing_and_exchange_of_gases.pdf",
        "cell.pdf", "cell_cycle_and_cell_division.pdf",
        "chemincal_coordination_and_integration.pdf",
        "excetory_products_and_their_coodination.pdf",
        "locomotion_and_movement.pdf", "morphology_of_flowering_plants.pdf",
        "neural_control_and_elimination.pdf", "photosynthesis_in_higher_plants.pdf",
        "plant_growth_and_development.pdf", "plant_kingdom.pdf",
        "respiration_in_plants.pdf", "structural_organization_in_animals.pdf",
        "the_living_world.pdf",
    ],
    "Chemistry": [
        "chemical_bonding_molecular_structure.pdf",
        "classification_of_elements_periodicity.pdf", "equilibrium.pdf",
        "hydrocarbons.pdf", "organic_chem_some_basic_principals.pdf",
        "redox_reactions.pdf", "some_basic_concepts.pdf",
        "structure_of_atom.pdf", "thermodynamics.pdf",
    ],
    "Physics": [
        "gravitation.pdf", "kinetic_theory.pdf", "laws_motion.pdf",
        "motion_in_plane.pdf", "motion_straight_line.pdf", "oscillations.pdf",
        "properties_of_fluids.pdf", "properties_of_solids.pdf",
        "rotational_motion.pdf", "thermal_properties.pdf",
        "thermodynamics.pdf", "units_measurements.pdf",
        "waves.pdf", "work_energy_power.pdf",
    ],
}

for subj in chapters:
    chapters[subj] = [clean_name(ch) for ch in chapters[subj]]

chapter = None
if subject:
    chapter = st.selectbox("Select Chapter", [""] + chapters[subject])

# PDF Upload
st.subheader("📄 Upload PDF Chapter")
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

# Configuration
col1, col2 = st.columns(2)
with col1:
    max_chunk_words = st.number_input("Max words per chunk", 100, 500, 400, 50)
with col2:
    overlap_words = st.number_input("Overlap words", 0, 100, 50, 10)

col3, col4 = st.columns(2)
with col3:
    skip_last_pages = st.number_input("Skip last N pages", 0, 20, 7, 1)
with col4:
    if model_type == "difficulty":
        difficulty_level = st.select_slider(
            "🎯 Question Difficulty",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: DIFFICULTY_LABELS[x],
        )
    else:
        difficulty_level = None
        st.info("ℹ️ Difficulty selector requires the difficulty model")

# Generate
if st.button("🔮 Generate Questions from PDF", type="primary"):
    if not subject or not chapter:
        st.error("Please select both subject and chapter.")
    elif not uploaded_file:
        st.error("Please upload a PDF file.")
    else:
        with st.spinner("📖 Extracting text..."):
            pdf_text = extract_text_from_pdf(uploaded_file, skip_last_pages)

        if pdf_text:
            total_words = len(pdf_text.split())
            st.success(f"✅ {total_words} words extracted (skipped last {skip_last_pages} pages)")

            with st.spinner("✂️ Chunking..."):
                chunks = chunk_text(pdf_text, max_chunk_words, overlap_words)
            st.info(f"📊 {len(chunks)} chunks")

            st.divider()
            if difficulty_level:
                st.subheader(f"📝 Generated Questions — {DIFFICULTY_LABELS[difficulty_level]}")
            else:
                st.subheader("📝 Generated Questions")

            progress = st.progress(0)
            status = st.empty()
            all_questions = []

            for idx, chunk in enumerate(chunks, 1):
                progress.progress(idx / len(chunks))
                status.text(f"Processing chunk {idx}/{len(chunks)}...")

                with st.expander(f"🔹 Chunk {idx} ({len(chunk.split())} words)", expanded=(idx == 1)):
                    st.caption("**Preview:**")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)

                    with st.spinner(f"Generating..."):
                        questions = generate_questions(chunk, difficulty_level)

                    relevance = calculate_relevance(chunk, questions)

                    st.markdown("**Generated Questions:**")
                    st.markdown(f"> {questions}")

                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.progress(relevance / 100)
                    with c2:
                        st.metric("Relevance", f"{relevance:.1f}%")

                    all_questions.append({
                        'chunk_id': idx, 'questions': questions, 'relevance': relevance
                    })

            progress.progress(100)
            status.text("✅ Done!")

            # Summary
            st.divider()
            st.subheader("📊 Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Chunks", len(chunks))
            with c2:
                st.metric("Words", total_words)
            with c3:
                avg_rel = sum(q['relevance'] for q in all_questions) / len(all_questions)
                st.metric("Avg Relevance", f"{avg_rel:.1f}%")

            # Download
            st.divider()
            st.subheader("💾 Download")
            dl = f"Questions for {subject} - {chapter}\n"
            if difficulty_level:
                dl += f"Difficulty: {DIFFICULTY_LABELS[difficulty_level]}\n"
            dl += "=" * 60 + "\n\n"
            for q in all_questions:
                dl += f"Chunk {q['chunk_id']} (Relevance: {q['relevance']:.1f}%)\n"
                dl += "-" * 60 + "\n"
                dl += f"{q['questions']}\n\n"

            st.download_button(
                "📥 Download All Questions",
                data=dl,
                file_name=f"{subject}_{chapter}_questions.txt",
                mime="text/plain",
            )

st.markdown("---")
st.caption("Developed using fine-tuned T5 + SentenceTransformer — © 2025 Harsh Saindane")
