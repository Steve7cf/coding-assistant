"""
Python Coding Assistant using Free ML Models
Setup with uv:
    uv pip install transformers torch

Or create a new project:
    uv init coding-assistant
    cd coding-assistant
    uv add transformers torch
    uv run python coding_assistant.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodingAssistant:
    def __init__(self, model_name="Salesforce/codegen-350M-mono"):
        """
        Initialize the coding assistant with a free code generation model.
        
        Available free models:
        - "Salesforce/codegen-350M-mono" (smaller, faster)
        - "Salesforce/codegen-2B-mono" (larger, better quality)
        - "bigcode/starcoderbase-1b" (good for multiple languages)
        """
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {self.device}!")
    
    def generate_code(self, prompt, max_length=150, temperature=0.7, num_return=1):
        """
        Generate code based on the given prompt.
        
        Args:
            prompt: The code description or partial code
            max_length: Maximum tokens to generate
            temperature: Creativity (0.1-1.0, lower is more deterministic)
            num_return: Number of completions to generate
        
        Returns:
            List of generated code strings
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=num_return,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_codes = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_codes
    
    def complete_code(self, code_snippet, max_new_tokens=100):
        """
        Complete a partial code snippet.
        
        Args:
            code_snippet: Incomplete code to complete
            max_new_tokens: Number of new tokens to generate
        
        Returns:
            Completed code string
        """
        inputs = self.tokenizer(code_snippet, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        completed_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completed_code
    
    def explain_code(self, code):
        """
        Generate an explanation for the given code (works best with instruction-tuned models).
        """
        prompt = f"# Explain this code:\n{code}\n# Explanation:"
        return self.generate_code(prompt, max_length=200, temperature=0.3)[0]


def main():
    # Initialize the assistant
    assistant = CodingAssistant()
    
    print("\n" + "="*60)
    print("Python Coding Assistant")
    print("="*60)
    
    # Example 1: Generate a function
    print("\n1. Generate a function to calculate fibonacci:")
    prompt1 = "def fibonacci(n):\n    \"\"\"\n    Calculate the nth fibonacci number\n    \"\"\"\n"
    result1 = assistant.complete_code(prompt1, max_new_tokens=150)
    print(result1)
    
    # Example 2: Complete code
    print("\n" + "-"*60)
    print("\n2. Complete a list comprehension:")
    prompt2 = "# Get all even numbers from a list\nnumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\neven_numbers = "
    result2 = assistant.complete_code(prompt2, max_new_tokens=50)
    print(result2)
    
    # Example 3: Generate from description
    print("\n" + "-"*60)
    print("\n3. Generate code from description:")
    prompt3 = "# Function to read a CSV file and return a pandas dataframe\nimport pandas as pd\n\ndef"
    result3 = assistant.complete_code(prompt3, max_new_tokens=100)
    print(result3)
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*60)
    
    while True:
        user_input = input("\nEnter your code prompt: ")
        if user_input.lower() == 'quit':
            break
        
        try:
            completion = assistant.complete_code(user_input, max_new_tokens=150)
            print("\nGenerated code:")
            print("-" * 40)
            print(completion)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()