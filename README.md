# Python Coding Assistant 🤖

A free, open-source AI coding assistant powered by Hugging Face Transformers. Generate, complete, and understand Python code using state-of-the-art language models - all running locally!

## Features

- **Code Generation**: Create functions from natural language descriptions
- **Code Completion**: Intelligently complete partial code snippets
- **Multiple Models**: Support for various free, open-source code models
- **GPU Acceleration**: Automatically uses GPU if available, falls back to CPU
- **Interactive Mode**: Chat-style interface for coding assistance
- **Privacy First**: Runs entirely on your machine, no data sent to external servers

## Installation

### Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone or create your project directory
mkdir coding-assistant
cd coding-assistant

# Install dependencies
uv pip install transformers torch

# Or add to a uv project
uv add transformers torch
```

### Using pip

```bash
pip install transformers torch
```

## Quick Start

```python
from coding_assistant import CodingAssistant

# Initialize the assistant
assistant = CodingAssistant()

# Generate code completion
prompt = "def calculate_average(numbers):\n    "
result = assistant.complete_code(prompt)
print(result)
```

## Usage

### Run the Interactive Assistant

```bash
uv run python coding_assistant.py
```

### Examples

#### 1. Generate a Function

```python
prompt = "def fibonacci(n):\n    \"\"\"\n    Calculate nth fibonacci number\n    \"\"\"\n"
code = assistant.complete_code(prompt, max_new_tokens=150)
```

#### 2. Complete Code Snippets

```python
prompt = "# Get all even numbers\nnumbers = [1,2,3,4,5]\neven = "
code = assistant.complete_code(prompt, max_new_tokens=50)
```

#### 3. Generate Multiple Solutions

```python
prompt = "def sort_dict_by_value(d):\n"
solutions = assistant.generate_code(prompt, num_return=3)
for i, solution in enumerate(solutions, 1):
    print(f"Solution {i}:\n{solution}\n")
```

## Available Models

The assistant supports multiple free models. Choose based on your needs:

| Model                          | Size | Speed  | Quality  | Best For                                |
| ------------------------------ | ---- | ------ | -------- | --------------------------------------- |
| `Salesforce/codegen-350M-mono` | 350M | ⚡⚡⚡ | ⭐⭐     | Quick completions, low-resource systems |
| `Salesforce/codegen-2B-mono`   | 2B   | ⚡⚡   | ⭐⭐⭐   | Better quality, needs more RAM          |
| `bigcode/starcoderbase-1b`     | 1B   | ⚡⚡   | ⭐⭐⭐   | Multi-language support                  |
| `replit/replit-code-v1-3b`     | 3B   | ⚡     | ⭐⭐⭐⭐ | Best quality, high-resource systems     |

### Changing Models

```python
# Use a different model
assistant = CodingAssistant(model_name="bigcode/starcoderbase-1b")
```

## Configuration

### Adjust Generation Parameters

```python
# More creative outputs
code = assistant.generate_code(
    prompt="def hello_world():",
    max_length=200,
    temperature=0.9,  # Higher = more creative
    num_return=1
)

# More deterministic outputs
code = assistant.complete_code(
    prompt="def factorial(n):",
    max_new_tokens=100,
    temperature=0.3  # Lower = more focused
)
```

### Parameters Explained

- **max_length**: Maximum total tokens (input + output)
- **max_new_tokens**: Maximum new tokens to generate
- **temperature**: Controls randomness (0.1-1.0)
  - Low (0.1-0.3): Deterministic, focused
  - Medium (0.4-0.7): Balanced
  - High (0.8-1.0): Creative, diverse
- **num_return**: Number of different completions to generate

## System Requirements

### Minimum

- Python 3.8+
- 4GB RAM
- CPU only

### Recommended

- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA support (for faster inference)

## Performance Tips

1. **Use GPU**: The assistant automatically detects and uses GPU if available
2. **Choose appropriate model**: Smaller models are faster but less accurate
3. **Adjust max_tokens**: Lower values = faster generation
4. **Lower temperature**: More deterministic = faster sampling

## Troubleshooting

### Out of Memory Error

```python
# Use a smaller model
assistant = CodingAssistant(model_name="Salesforce/codegen-350M-mono")
```

### Slow Generation

- Reduce `max_new_tokens` or `max_length`
- Use a smaller model
- Lower the `temperature`

### Import Errors

```bash
# Reinstall dependencies
uv pip install --force-reinstall transformers torch
```

## Project Structure

```
coding-assistant/
├── coding_assistant.py    # Main assistant code
├── README.md             # This file
├── pyproject.toml        # Project configuration (optional)
└── examples/             # Usage examples (optional)
```

## Contributing

Contributions are welcome! Some ideas:

- Add support for more programming languages
- Implement code explanation features
- Create a web UI with Gradio/Streamlit
- Add code quality analysis
- Implement caching for faster repeated queries

## License

This project uses models and libraries with their respective licenses:

- Transformers: Apache 2.0
- PyTorch: BSD-style license
- Individual models: Check their model cards on Hugging Face

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Salesforce](https://github.com/salesforce/CodeGen) for CodeGen models
- [BigCode](https://www.bigcode-project.org/) for StarCoder models
- [Replit](https://replit.com/) for their code models

## Resources

- [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Code Generation Guide](https://huggingface.co/docs/transformers/tasks/code_generation)

## Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review Hugging Face model documentation
3. Open an issue with your error message and system info

---

**Note**: First run will download the model (350MB-6GB depending on model choice). Subsequent runs will use cached models.
