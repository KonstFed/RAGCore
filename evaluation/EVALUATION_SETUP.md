# Evaluation Setup Guide

## Prerequisites

1. **Install dependencies** using `uv`:
   ```bash
   uv sync
   ```

2. **Start Qdrant vector database**:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

3. **Configure API keys** in .env

## Data Structure

```
evaluation/swe_qa_data/
├── repos.txt          # Repository URLs with commit hashes (format: URL commit_hash)
└── questions/         # Directory with question files
    ├── astropy.jsonl
    ├── django.jsonl
    └── ...            # One .jsonl file per repository
```

## Running Evaluation

Execute the evaluation script with required arguments:

```bash
uv run python evaluation/eval_swe_qa.py \
    --repo_meta evaluation/swe_qa_data/repos.txt \
    --question_path evaluation/swe_qa_data/questions \
    --output result_qa.jsonl
```

### Arguments

- `--repo_meta`: Path to file containing repository URLs and commit hashes
- `--question_path`: Path to directory containing question JSONL files
- `--output`: Path to output JSONL file

## What It Does

1. **Indexes repositories**: Downloads and indexes all repositories listed in `repos.txt`
2. **Processes questions**: For each repository, runs all questions through the RAG pipeline
3. **Generates answers**: Produces answers using the configured search engine and LLM
4. **Saves results**: Writes question-answer pairs to the output JSONL file

## Output Format

The output file contains JSON Lines with the following structure:
```json
{"question": "...", "answer": "..."}
{"question": "...", "answer": "..."}
```

## Notes

- Evaluation runs sequentially (no parallelization currently)
- Processing time depends on the number of repositories and questions
- Ensure sufficient disk space for repository downloads and Qdrant storage

