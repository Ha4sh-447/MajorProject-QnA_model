#!/usr/bin/env python3
"""
Bloom's Taxonomy Question Generation — Streamlit App

Generates questions at 6 Bloom's taxonomy levels from uploaded PDFs.
Uses sampling-based generation for novel, creative output.
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

BLOOMS = {
    1: ("Remember",   "Recall facts & definitions"),
    2: ("Understand", "Explain & interpret concepts"),
    3: ("Apply",      "Use knowledge in new situations"),
    4: ("Analyze",    "Find relationships & patterns"),
    5: ("Evaluate",   "Judge & critique approaches"),
    6: ("Create",     "Design & propose new ideas"),
}

MODEL_DIR = "../../question_gen_blooms_model"
BASE_MODEL = "google/flan-t5-small"
FALLBACK_DIR = "../../question_gen_model_final"


@st.cache_resource
def load_models():
    try:
        if not HAS_PEFT:
            raise ImportError("peft not installed")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base = AutoModelForSeq2SeqLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16,
        ).to(device)
        model = PeftModel.from_pretrained(base, MODEL_DIR)
        model.eval()
        model_type = "blooms"
    except Exception:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained(FALLBACK_DIR)
        model = T5ForConditionalGeneration.from_pretrained(FALLBACK_DIR)
        model_type = "basic"

    semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model, semantic_model, model_type


tokenizer, model, semantic_model, model_type = load_models()


def extract_text_from_pdf(pdf_file, skip_last=7):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        pages = len(reader.pages) - skip_last
        if pages <= 0:
            st.error(f"PDF too short to skip {skip_last} pages")
            return None
        return "\n".join(reader.pages[i].extract_text() for i in range(pages)).strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def chunk_text(text, max_words=400, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        i += max_words - overlap
        if i + overlap >= len(words):
            break
    return chunks


def generate_questions(chunk, level=None):
    if level and model_type == "blooms":
        name = BLOOMS[level][0].lower()
        prompt = f"generate a {name} level question: {chunk}"
    else:
        prompt = f"generate questions: {chunk}"

    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    out = model.generate(
        inputs, max_length=384, min_length=20,
        do_sample=True, temperature=0.8, top_p=0.92, top_k=50,
        no_repeat_ngram_size=3, repetition_penalty=1.3,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def relevance(chunk, question):
    e1 = semantic_model.encode(chunk, convert_to_tensor=True)
    e2 = semantic_model.encode(question, convert_to_tensor=True)
    return float(util.cos_sim(e1, e2)[0][0]) * 100


def clean_name(f):
    return f.replace(".pdf", "").replace("_", " ").strip().title()


# ─── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Bloom's Q Generator", page_icon="🌸", layout="wide")
st.title("🌸 Bloom's Taxonomy Question Generator")
st.markdown("Generate questions at 6 cognitive levels from PDF chapters.")

if model_type == "blooms":
    st.success("✅ Bloom's taxonomy model loaded")
else:
    st.warning("⚠️ Fallback model — Bloom's levels unavailable")

st.divider()

st.subheader("Class: 11 (Fixed)")
subject = st.selectbox("Subject", ["", "Biology", "Chemistry", "Physics"])

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
for s in chapters:
    chapters[s] = [clean_name(c) for c in chapters[s]]

chapter = None
if subject:
    chapter = st.selectbox("Chapter", [""] + chapters[subject])

st.subheader("📄 Upload PDF")
uploaded = st.file_uploader("Choose PDF", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    max_words = st.number_input("Max words/chunk", 100, 500, 400, 50)
with col2:
    overlap = st.number_input("Overlap words", 0, 100, 50, 10)

col3, col4 = st.columns(2)
with col3:
    skip_pages = st.number_input("Skip last N pages", 0, 20, 7, 1)
with col4:
    if model_type == "blooms":
        level = st.select_slider(
            "🌸 Bloom's Level",
            options=[1, 2, 3, 4, 5, 6],
            value=2,
            format_func=lambda x: f"L{x} — {BLOOMS[x][0]} ({BLOOMS[x][1]})",
        )
    else:
        level = None
        st.info("ℹ️ Requires Bloom's model")

if st.button("🔮 Generate Questions", type="primary"):
    if not subject or not chapter:
        st.error("Select subject and chapter.")
    elif not uploaded:
        st.error("Upload PDF.")
    else:
        with st.spinner("📖 Extracting..."):
            text = extract_text_from_pdf(uploaded, skip_pages)

        if text:
            words = len(text.split())
            st.success(f"✅ {words} words extracted")

            with st.spinner("✂️ Chunking..."):
                chunks = chunk_text(text, max_words, overlap)
            st.info(f"📊 {len(chunks)} chunks")

            st.divider()
            if level:
                st.subheader(f"📝 Questions — L{level} {BLOOMS[level][0]}: {BLOOMS[level][1]}")
            else:
                st.subheader("📝 Generated Questions")

            progress = st.progress(0)
            status = st.empty()
            all_qs = []

            for idx, chunk in enumerate(chunks, 1):
                progress.progress(idx / len(chunks))
                status.text(f"Chunk {idx}/{len(chunks)}...")

                with st.expander(f"🔹 Chunk {idx} ({len(chunk.split())} words)", expanded=(idx == 1)):
                    st.caption("**Preview:**")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)

                    with st.spinner("Generating..."):
                        qs = generate_questions(chunk, level)

                    rel = relevance(chunk, qs)

                    st.markdown("**Generated:**")
                    st.markdown(f"> {qs}")

                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.progress(rel / 100)
                    with c2:
                        st.metric("Relevance", f"{rel:.1f}%")

                    all_qs.append({"id": idx, "questions": qs, "relevance": rel})

            progress.progress(100)
            status.text("✅ Done!")

            st.divider()
            st.subheader("📊 Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Chunks", len(chunks))
            with c2:
                st.metric("Words", words)
            with c3:
                avg = sum(q["relevance"] for q in all_qs) / len(all_qs)
                st.metric("Avg Relevance", f"{avg:.1f}%")

            st.divider()
            dl = f"Bloom's Questions — {subject} - {chapter}\n"
            if level:
                dl += f"Level: {level} ({BLOOMS[level][0]})\n"
            dl += "=" * 60 + "\n\n"
            for q in all_qs:
                dl += f"Chunk {q['id']} (Relevance: {q['relevance']:.1f}%)\n"
                dl += "-" * 60 + f"\n{q['questions']}\n\n"

            st.download_button("📥 Download", dl,
                               f"{subject}_{chapter}_blooms_questions.txt", "text/plain")

st.markdown("---")
st.caption("Bloom's Taxonomy QG — Fine-tuned T5 + SentenceTransformer — © 2025 Harsh Saindane")
