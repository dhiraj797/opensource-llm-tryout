# Insurance Claims LLM Chatbot

An experimental Python project that explores the use of open-source large language models (LLMs) for automating insurance claims processing and policy Q&A.

## Overview

This project demonstrates how open-source LLMs can be applied to domain-specific use cases in the insurance industry — including claims intake, policy lookups, and conversational customer support.

## Features

- Conversational chatbot interface for general insurance queries
- Claims-specific chatbot with domain knowledge
- Policy data management and retrieval
- Model quantization for efficient local inference
- Weight visualisation utilities for model analysis

## Project Structure

```
├── chat.py               # General-purpose chatbot interface
├── claims_chatbot.py     # Insurance claims chatbot logic
├── policy_data.py        # Policy data handling
├── quantize_model.py     # Model quantization utilities
├── visualize_weights.py  # Weight visualisation tools
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/dhiraj797/opensource-llm-tryout.git
cd opensource-llm-tryout
pip install -r requirements.txt
```

### Usage

```bash
python chat.py
```

## Tech Stack

- Python
- Open-source LLMs (HuggingFace / Ollama)
- Model quantization tools

## License

MIT
