import pandas as pd
import re
from transformers import AutoTokenizer

def extract_chapter_metadata(input_text):
    """Extract chapter title, topic, and key concepts from input"""
    lines = input_text.split('\n')
    metadata = {
        'chapter_title': '',
        'topic': '',
        'section': ''
    }
    
    # Look for chapter/section headers in first few lines
    for line in lines[:10]:
        line = line.strip()
        # Look for patterns like "2.3.2 Particle Nature..." or "Chapter 2: ..."
        if re.match(r'\d+\.\d+', line) or 'Chapter' in line:
            if len(line) < 200:  # Likely a header, not content
                if not metadata['chapter_title']:
                    metadata['chapter_title'] = line
                elif not metadata['section']:
                    metadata['section'] = line
    
    return metadata

def create_contextual_prefix(metadata, is_continuation=False):
    """Create a context-preserving prefix for chunks"""
    prefix_parts = []
    
    if metadata['chapter_title']:
        prefix_parts.append(f"Chapter: {metadata['chapter_title']}")
    
    if metadata['section']:
        prefix_parts.append(f"Section: {metadata['section']}")
    
    if is_continuation:
        prefix_parts.append("(Continued from previous section)")
    
    if prefix_parts:
        return " | ".join(prefix_parts) + "\n\n"
    return ""

def smart_split_text(text, tokenizer, max_tokens=400, overlap_tokens=80):
    """
    Split text intelligently at paragraph/sentence boundaries while maintaining context
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    
    # Split into paragraphs first
    paragraphs = text.split('\n\n')
    
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_tokens = tokenizer.encode(para, add_special_tokens=False)
        para_token_count = len(para_tokens)
        
        # If single paragraph is too long, split by sentences
        if para_token_count > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
                sent_token_count = len(sent_tokens)
                
                if current_tokens + sent_token_count > max_tokens:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        
                        # Keep last sentence for context (overlap)
                        if len(current_chunk) > 0:
                            overlap_text = current_chunk[-1]
                            overlap_token_count = len(tokenizer.encode(overlap_text, add_special_tokens=False))
                            
                            if overlap_token_count <= overlap_tokens:
                                current_chunk = [overlap_text, sent]
                                current_tokens = overlap_token_count + sent_token_count
                            else:
                                current_chunk = [sent]
                                current_tokens = sent_token_count
                        else:
                            current_chunk = [sent]
                            current_tokens = sent_token_count
                else:
                    current_chunk.append(sent)
                    current_tokens += sent_token_count
        
        # Normal paragraph handling
        elif current_tokens + para_token_count > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                # Keep last paragraph for context (overlap)
                if len(current_chunk) > 0:
                    overlap_text = current_chunk[-1]
                    overlap_token_count = len(tokenizer.encode(overlap_text, add_special_tokens=False))
                    
                    if overlap_token_count <= overlap_tokens:
                        current_chunk = [overlap_text, para]
                        current_tokens = overlap_token_count + para_token_count
                    else:
                        current_chunk = [para]
                        current_tokens = para_token_count
                else:
                    current_chunk = [para]
                    current_tokens = para_token_count
        else:
            current_chunk.append(para)
            current_tokens += para_token_count
    
    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def extract_and_group_questions(target_text):
    """
    Extract questions and group them by topic/concept for better context preservation
    """
    questions = []
    lines = target_text.split('\n')
    
    current_question = []
    current_topic = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect question start patterns
        is_new_question = (
            re.match(r'^\d+[\.)]\s*', line) or  # "1. " or "1) "
            re.match(r'^Q\d+', line) or          # "Q1"
            re.match(r'^Question \d+', line) or  # "Question 1"
            (line.endswith('?') and len(current_question) > 0)  # New question ending with ?
        )
        
        # Detect topic hints (like "Calculate", "Explain", "Define")
        topic_keywords = ['calculate', 'explain', 'define', 'describe', 'what', 'why', 'how']
        if any(keyword in line.lower() for keyword in topic_keywords):
            topic_hint = next((kw for kw in topic_keywords if kw in line.lower()), None)
            if topic_hint and not current_topic:
                current_topic = topic_hint
        
        if is_new_question and current_question:
            questions.append({
                'text': '\n'.join(current_question),
                'topic': current_topic
            })
            current_question = [line]
            current_topic = None
        else:
            current_question.append(line)
    
    # Add last question
    if current_question:
        questions.append({
            'text': '\n'.join(current_question),
            'topic': current_topic
        })
    
    return questions

def match_questions_to_chunks(chunks, questions, tokenizer, max_target_tokens=120):
    """
    Intelligently match questions to relevant chunks based on content similarity
    """
    chunk_question_pairs = []
    
    # Extract key terms from each chunk for matching
    def get_key_terms(text):
        # Simple keyword extraction (you can enhance this)
        words = re.findall(r'\b[A-Z][a-z]+\b|\b(?:electron|photon|atom|energy|frequency|wavelength|quantum|radiation|spectrum)\b', text.lower())
        return set(words)
    
    chunk_terms = [get_key_terms(chunk) for chunk in chunks]
    
    # Distribute questions across chunks based on relevance
    used_questions = set()
    
    for i, chunk in enumerate(chunks):
        chunk_keywords = chunk_terms[i]
        matched_questions = []
        
        # Find questions that relate to this chunk
        for j, q_data in enumerate(questions):
            if j in used_questions:
                continue
            
            q_text = q_data['text']
            q_keywords = get_key_terms(q_text)
            
            # Check for keyword overlap
            overlap = len(chunk_keywords & q_keywords)
            
            if overlap > 0 or i == 0:  # First chunk gets unmatched questions
                matched_questions.append((j, q_data))
                used_questions.add(j)
                
                # Limit questions per chunk based on token budget
                current_tokens = sum(
                    len(tokenizer.encode(q['text'], add_special_tokens=False)) 
                    for _, q in matched_questions
                )
                
                if current_tokens >= max_target_tokens:
                    break
        
        chunk_question_pairs.append((i, matched_questions))
    
    # Distribute remaining questions to last chunks
    remaining_questions = [q for j, q in enumerate(questions) if j not in used_questions]
    if remaining_questions and chunk_question_pairs:
        last_idx = len(chunk_question_pairs) - 1
        chunk_question_pairs[last_idx] = (
            last_idx,
            chunk_question_pairs[last_idx][1] + [(len(questions), q) for q in remaining_questions]
        )
    
    return chunk_question_pairs

def create_chunked_dataset(input_path, output_path, tokenizer):
    """
    Create context-aware chunked dataset with intelligent question matching
    """
    data = pd.read_json(input_path)
    new_rows = []
    
    stats = {
        'original_rows': len(data),
        'chunks_created': 0,
        'avg_chunks_per_row': 0,
        'questions_preserved': 0,
        'total_questions': 0
    }
    
    print("Processing dataset with context-aware chunking...\n")
    
    for idx, row in data.iterrows():
        input_text = row['input_text']
        target_text = row['target_text']
        
        # Extract metadata for context
        content = input_text.replace(
            "Generate exam-style questions from the following text:\n", ""
        ).strip()
        
        metadata = extract_chapter_metadata(content)
        
        # Extract and categorize questions
        questions = extract_and_group_questions(target_text)
        stats['total_questions'] += len(questions)
        
        # Check if chunking is needed
        input_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        target_tokens = len(tokenizer.encode(target_text, add_special_tokens=False))
        
        # If content is reasonable size, keep as is
        if input_tokens <= 450 and target_tokens <= 120:
            new_rows.append({
                'input_text': input_text,
                'target_text': target_text
            })
            stats['chunks_created'] += 1
            stats['questions_preserved'] += len(questions)
        
        # If only target is too long, keep full context but limit questions
        elif input_tokens <= 450 and target_tokens > 120:
            # Take questions in groups
            questions_per_group = 3
            for i in range(0, len(questions), questions_per_group):
                q_group = questions[i:i + questions_per_group]
                q_texts = '\n'.join([q['text'] for q in q_group])
                
                new_rows.append({
                    'input_text': input_text,
                    'target_text': q_texts
                })
                stats['chunks_created'] += 1
                stats['questions_preserved'] += len(q_group)
        
        # If input is too long, do intelligent chunking
        else:
            # Create context-aware chunks
            chunks = smart_split_text(content, tokenizer, max_tokens=420, overlap_tokens=80)
            
            # Match questions to relevant chunks
            chunk_question_pairs = match_questions_to_chunks(
                chunks, questions, tokenizer, max_target_tokens=120
            )
            
            for chunk_idx, matched_questions in chunk_question_pairs:
                if not matched_questions:
                    continue
                
                chunk = chunks[chunk_idx]
                
                # Add context prefix
                context_prefix = create_contextual_prefix(
                    metadata, 
                    is_continuation=(chunk_idx > 0)
                )
                
                # Combine questions
                q_texts = '\n'.join([q['text'] for _, q in matched_questions])
                
                # Create training example
                new_rows.append({
                    'input_text': f"Generate exam-style questions from the following text:\n{context_prefix}{chunk}",
                    'target_text': q_texts
                })
                
                stats['chunks_created'] += 1
                stats['questions_preserved'] += len(matched_questions)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(data)} rows...")
    
    # Create dataframe
    new_df = pd.DataFrame(new_rows)
    
    # Validate lengths
    print("\n" + "="*60)
    print("VALIDATION & STATISTICS")
    print("="*60)
    
    new_df['input_tokens'] = new_df['input_text'].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )
    new_df['target_tokens'] = new_df['target_text'].apply(
        lambda x: len(tokenizer.encode(x, add_special_tokens=False))
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Original rows: {stats['original_rows']}")
    print(f"  New rows (chunks): {stats['chunks_created']}")
    print(f"  Expansion factor: {stats['chunks_created'] / stats['original_rows']:.2f}x")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  Questions preserved: {stats['questions_preserved']}")
    print(f"  Preservation rate: {stats['questions_preserved'] / stats['total_questions'] * 100:.1f}%")
    
    print(f"\n📏 Token Length Statistics:")
    print(f"\nInput tokens:")
    print(f"  Mean: {new_df['input_tokens'].mean():.0f}")
    print(f"  Max: {new_df['input_tokens'].max()}")
    print(f"  Min: {new_df['input_tokens'].min()}")
    print(f"  > 512: {(new_df['input_tokens'] > 512).sum()} rows")
    
    print(f"\nTarget tokens:")
    print(f"  Mean: {new_df['target_tokens'].mean():.0f}")
    print(f"  Max: {new_df['target_tokens'].max()}")
    print(f"  Min: {new_df['target_tokens'].min()}")
    print(f"  > 150: {(new_df['target_tokens'] > 150).sum()} rows")
    
    # Show sample
    print(f"\n📝 Sample Output (first chunk):")
    print("-" * 60)
    sample = new_df.iloc[0]
    print(f"Input (first 300 chars):\n{sample['input_text'][:300]}...")
    print(f"\nTarget:\n{sample['target_text'][:400]}...")
    print("-" * 60)
    
    # Remove validation columns
    new_df = new_df.drop(['input_tokens', 'target_tokens'], axis=1)
    
    # Save
    new_df.to_json(output_path, orient='records', indent=2)
    print(f"\n✅ Saved context-aware chunked dataset to: {output_path}")
    print("="*60 + "\n")
    
    return new_df

# Run preprocessing
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

print("\nStarting context-aware chunking...\n")
df = create_chunked_dataset(
    "./dataset/final_preprocessed_dataset.json",
    "./dataset/chunked_dataset.json",
    tokenizer
)

print("Done! You can now train with: python train_final.py")
