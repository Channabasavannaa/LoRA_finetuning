def prompt_template(text_chunk: str, num_records: int):
    return f"""
You are a data generation model. Read the following medical text chunk and generate exactly {num_records} high-quality question-and-answer pairs.

### Rules:
- Only generate a JSON object.
- JSON MUST follow this format:

{{
  "generated": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}

- The number of records MUST be {num_records}.
- Questions must be clear, specific, and based ONLY on the provided text.
- Answers must be accurate and taken strictly from the text.
- Do NOT invent facts not present in the text.
- DO NOT include markdown, explanations, or extra text — only JSON output.

### Text Chunk:
\"\"\"{text_chunk}\"\"\"
"""