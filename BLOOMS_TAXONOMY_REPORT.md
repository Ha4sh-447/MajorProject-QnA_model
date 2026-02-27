# Technical Report: Redesigning Bloom's Taxonomy Question Generation

**Date**: February 27, 2026  
**Project**: AI-Driven Educational Assessment  
**Author**: Antigravity AI (Pair-programmed with harsh)

---

## 1. Executive Summary
This project aimed to refine an AI question generation (QGen) system aligned with **Bloom’s Taxonomy**. The initial model suffered from severe data imbalance, leading to the hallucination of non-existent tags (e.g., `[ELECTRATE]`) and an inability to generate higher-order thinking questions (Evaluate & Create).

Through two parallel redesign strategies—**Option A (6-Level Augmentation)** and **Option B (4-Level Consolidation)**—we successfully eliminated tagging errors and reached **100% tag accuracy**. This report details the pedagogical, architectural, and data engineering decisions that enabled this breakthrough.

---

## 2. Theoretical Framework: Bloom’s Taxonomy
Bloom’s Taxonomy is a hierarchical model used to classify educational learning objectives into levels of complexity and specificity.

### The Six Levels
1.  **Remember (L1)**: Recalling facts and basic concepts (e.g., *Define..., List...*)
2.  **Understand (L2)**: Explaining ideas or concepts (e.g., *Classify..., Discuss...*)
3.  **Apply (L3)**: Using information in new situations (e.g., *Calculate..., Solve...*)
4.  **Analyze (L4)**: Drawing connections among ideas (e.g., *Differentiate..., Compare...*)
5.  **Evaluate (L5)**: Justifying a stand or decision (e.g., *Appraise..., Judge...*)
6.  **Create (L6)**: Producing new or original work (e.g., *Design..., Develop...*)

---

## 3. The Problem: Root Cause Analysis
Our baseline model failed primarily due to **Dataset skew** and **Heuristic limitations**:

*   **Imbalance**: 53% of our data was L1 (Remember). L2 (Understand) and L6 (Create) were <5%. The model defaulted to the majority class whenever it was uncertain.
*   **Tag Hallucination**: With only ~9% of data for L5 (Evaluate), the model never learned the specific token `[EVALUATE]` reliably, resulting in gibberish tags like `[ELECTRATE]`.
*   **Precision Instability**: Initial attempts at 8-bit quantization and FP16 mixed precision caused `NaN` (Not a Number) gradients, breaking the training process.

---

## 4. The Engineering Journey: Precision & Stability

One of the most complex parts of this project was finding the right numerical precision for training a small model (`flan-t5-small`, 60M parameters) on standard hardware.

### Phase 1: 8-bit Quantization (Failed)
*   **Attempt**: Used `bitsandbytes` to load the model in 8-bit to save VRAM.
*   **Result**: Training loss was 0.0 or NaN from Step 1.
*   **Why**: 8-bit quantization works best for large models (7B+ parameters). For 60M parameters, the "rounding errors" are too aggressive, destroying the small weight matrices and causing the gradients to collapse.

### Phase 2: FP16 & AMP (Failed)
*   **Attempt**: Used `fp16=True` (Automatic Mixed Precision - AMP).
*   **Result**: Loss underflow followed by NaN.
*   **Why**: AMP converts weights to 16-bit for speed and back to 32-bit for loss calculation. Our loss values were so small that they "underflowed" to 0.0 in 16-bit. When the `GradScaler` tried to fix this, it produced inf/NaN values.

### Phase 3: Pure FP32 (Success)
*   **Resolution**: Since `flan-t5-small` only takes ~900MB of VRAM in full 32-bit precision, we disabled all quantization and mixed precision.
*   **Outcome**: Training was perfectly stable. Stable gradients allowed us to increase the **Learning Rate (5e-5)** and **LoRA Rank (r=32)**, giving the model enough capacity to learn all 6 tags.

---

## 5. Dataset Redesign Strategies

### Option A: 6-Level Augmentation
We kept all 6 levels but "forced" balance through synthetic data:
*   **Downsampling**: Reduced L1 from 1804 examples to 700.
*   **Key-Sentence Augmentation**: Instead of copying questions, we extracted factual sentences from the chapter text and used "Bloom's Verb Templates" to create target questions. 
    *   *Example*: From a sentence about Mitosis, we generated a Create-level prompt: *"Design an experiment to demonstrate..."*

### Option B: 4-Level Consolidation
We merged cognitive neighbors to create more robust categories:
1.  **Remember & Understand** (L1+L2)
2.  **Apply** (L3)
3.  **Analyze** (L4)
4.  **Evaluate & Create** (L5+L6)
*   **Result**: This significantly reduced "confusion" at the boundaries (e.g., whether a question is Evaluate vs Create), leading to much more diverse outputs (lower Self-BLEU).

---

## 6. Model Parameters & LoRA
We utilized **LoRA (Low-Rank Adaptation)** to fine-tune the model efficiently.

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **LoRA Rank (r)** | 32 | Higher than usual (8) to capture 6 distinct tag behaviors. |
| **Alpha** | 64 | Scaling factor for the learned weights. |
| **Target Modules** | q, v, k, o | Targeting all attention matrices to maximize cognitive shift. |
| **Epochs** | 10 | Enough for the model to see augmented samples multiple times. |
| **Max Length** | 512 / 128 | Input (source text) vs Output (question). |

---

## 7. Metrics Explained
To scientifically prove the model's improvement, we implemented 6 metrics:

1.  **BLEU (Bilingual Evaluation Understudy)**: Measures exact word overlap. Our score (~0.20) is healthy for creative tasks where we don't want the model to be a "copy-paste" machine.
2.  **ROUGE-L**: Measures the longest common subsequence. Captures the structural flow of the question.
3.  **BERTScore**: Uses contextual embeddings to measure semantic similarity. This is our most trusted quality metric; a score of **0.86** indicates very high relevance to the source text.
4.  **Tag Accuracy**: The percentage of outputs containing the correct `[TAG]`. We moved from **~60% to 100%**.
5.  **Self-BLEU**: Measures internal diversity. The **4-Level model (0.47)** produced more varied questions than the **6-Level model (0.52)**.
6.  **Level Distinctness**: A measure of how different a "Remember" question looks from an "Evaluate" question.

---

## 8. Conclusion & Recommendation
The **4-level model (Option B)** is the recommended path for production-grade applications. While the 6-level model is a significant technical achievement, the 4-level model offers a better balance of **diversity, reliability, and cognitive clarity**. 

The transition to **Full FP32 training** was the critical turning point that solved the stability issues, proving that for small models, "less is more" when it comes to optimization tricks.

---
*End of Report*
