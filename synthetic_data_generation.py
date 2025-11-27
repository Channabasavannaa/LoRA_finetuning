import pandas as pd
from colorama import Fore
import textwrap
import json
from typing import List
from pydantic import BaseModel
from litellm import completion
#from generated_prompt import prompt_template

# Function to create chunks of text
def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# Dummy enrichment function (replace with real enrichment if needed)
def enrich_text(text):
    # Example: just return text for now; you can add NLP enrichment here
    return text

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    stream = completion(
        model="ollama_chat/qwen2.5:14b",
        messages=[
            {
                "role": "user",
                "content": prompt_template(data, num_records),
            }
        ],
        stream=True,
        options={"num_predict": 2000},
        format=Response.model_json_schema(),
    )
    data = ""
    for x in stream: 
        delta = x['choices'][0]["delta"]["content"]
        if delta is not None: 
            print(Fore.LIGHTBLUE_EX+ delta + Fore.RESET, end="") 
            data += delta 
    return json.loads(data)

if __name__ == "__main__":
    # Path to the existing chunks file
    input_path = r"D:\LoRA_finetuning\LoRA_finetuning\Data\medquad_chunks.json"
    output_path = r"D:\LoRA_finetuning\LoRA_finetuning\Data\medquad_qa.json"

    # Load existing chunks
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    qa_data = []

    for item in chunks:
        raw = item.get("raw_text", "")

        # Simple split: everything before first '?' = question, rest = answer
        if "?" in raw:
            q, rest = raw.split("?", 1)
            question = q.strip() + "?"
            answer = rest.strip()
        else:
            question = raw.strip()
            answer = ""

        qa_data.append({
            "question": question,
            "answer": answer
        })

    # Save Q&A JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=4)

    print(Fore.GREEN + f"\nQ&A data saved to {output_path}" + Fore.RESET)