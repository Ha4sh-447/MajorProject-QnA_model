#!/usr/bin/env python3
"""
Comprehensive model testing script
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path="./question_gen_model_stable"):
    """Load the fine-tuned model"""
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small",
        device_map="auto"
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✅ Model loaded successfully\n")
    return model, tokenizer

def generate_questions(model, tokenizer, text, max_length=150, num_beams=4, temperature=0.7):
    """Generate questions from input text"""
    
    # Prepare input
    if not text.startswith("Generate"):
        text = f"Generate exam-style questions from the following text:\n{text}"
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,  # Prevent repetition
            repetition_penalty=1.2,
        )
    
    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

def test_multiple_samples(model, tokenizer):
    """Test model on various inputs"""
    
    test_cases = [
        {
            "name": "Physics - Photoelectric Effect",
            "text": """The photoelectric effect was explained by Einstein in 1905. When electromagnetic 
radiation of sufficient frequency strikes a metal surface, electrons are emitted. The kinetic 
energy of the emitted electrons depends on the frequency of the light, not its intensity. 
The minimum frequency required to eject electrons is called the threshold frequency."""
        },
        {
            "name": "Chemistry - Atomic Structure",
            "text": """Atoms consist of a nucleus containing protons and neutrons, surrounded by electrons 
in orbits. The atomic number is equal to the number of protons. Electrons occupy specific 
energy levels or shells around the nucleus. The first shell can hold 2 electrons, the second 
can hold 8 electrons."""
        },
        {
            "name": "Math - Quadratic Equations",
            "text": """A quadratic equation is a second-degree polynomial equation in the form ax² + bx + c = 0, 
where a, b, and c are constants and a ≠ 0. The solutions can be found using the quadratic formula: 
x = (-b ± √(b²-4ac)) / 2a. The discriminant b²-4ac determines the nature of the roots."""
        },
        {
            "name": "Biology - Cell Structure",
            "text": """The cell is the basic unit of life. Plant cells have a cell wall made of cellulose, 
while animal cells do not. Both types contain a cell membrane, cytoplasm, and nucleus. 
Mitochondria are the powerhouses of the cell, producing energy through cellular respiration."""
        },
        {
            "name": "Short Text Test",
            "text": """Newton's first law states that an object at rest stays at rest unless acted upon by an external force."""
        }
    ]
    
    print("="*80)
    print("TESTING MODEL ON VARIOUS INPUTS")
    print("="*80 + "\n")
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 80)
        print(f"Input text ({len(test['text'])} chars):")
        print(test['text'][:200] + "..." if len(test['text']) > 200 else test['text'])
        print("\nGenerated Questions:")
        
        try:
            questions = generate_questions(model, tokenizer, test['text'])
            print(questions)
            
            # Check for repetition
            words = questions.split()
            unique_ratio = len(set(words)) / len(words) if words else 0
            
            if unique_ratio < 0.5:
                print("\n⚠️  Warning: High repetition detected (unique ratio: {:.2f})".format(unique_ratio))
            else:
                print("\n✅ Output looks reasonable (unique ratio: {:.2f})".format(unique_ratio))
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "="*80 + "\n")

def interactive_mode(model, tokenizer):
    """Interactive testing mode"""
    print("="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your text (or 'quit' to exit):\n")
    
    while True:
        text = input("📝 Your text: ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text.strip():
            print("Please enter some text.\n")
            continue
        
        print("\n🤖 Generating questions...\n")
        questions = generate_questions(model, tokenizer, text)
        print(questions)
        print("\n" + "-"*80 + "\n")

def compare_with_baseline(model, tokenizer):
    """Compare fine-tuned model with base model"""
    print("="*80)
    print("COMPARING FINE-TUNED VS BASE MODEL")
    print("="*80 + "\n")
    
    test_text = """The Bohr model of the atom was proposed by Niels Bohr in 1913. 
According to this model, electrons orbit the nucleus in fixed circular paths called orbits 
or shells. Each orbit has a specific energy level. Electrons can jump from one orbit to 
another by absorbing or emitting energy."""
    
    print("Test text:")
    print(test_text)
    print("\n" + "-"*80)
    
    # Fine-tuned model
    print("\n1. FINE-TUNED MODEL OUTPUT:")
    print("-"*80)
    ft_output = generate_questions(model, tokenizer, test_text)
    print(ft_output)
    
    # Base model
    print("\n2. BASE MODEL OUTPUT (no fine-tuning):")
    print("-"*80)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small",
        device_map="auto"
    )
    base_output = generate_questions(base_model, tokenizer, test_text)
    print(base_output)
    
    print("\n" + "="*80 + "\n")

def main():
    """Main testing function"""
    import sys
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nMake sure the model directory exists: ./question_gen_model_stable")
        return
    
    # Menu
    print("Choose testing mode:")
    print("1. Test on predefined samples")
    print("2. Interactive mode")
    print("3. Compare with baseline model")
    print("4. All of the above")
    
    choice = input("\nYour choice (1-4): ").strip()
    
    if choice == "1":
        test_multiple_samples(model, tokenizer)
    elif choice == "2":
        interactive_mode(model, tokenizer)
    elif choice == "3":
        compare_with_baseline(model, tokenizer)
    elif choice == "4":
        test_multiple_samples(model, tokenizer)
        compare_with_baseline(model, tokenizer)
        interactive_mode(model, tokenizer)
    else:
        print("Invalid choice. Running all tests...")
        test_multiple_samples(model, tokenizer)

if __name__ == "__main__":
    main()
