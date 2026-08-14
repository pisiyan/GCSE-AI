import sys
import os
import json
import re
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, '.')
from user_data_ingestion import UserDataIngestion
from llm_client import LLMClient

ingestion = UserDataIngestion(llm_client=LLMClient())
docs = ingestion.convert_documents_to_text()
ans_doc = [d for d in docs if 'answer' in d['filename'].lower()][0]

prompt_path = 'prompts/extract_document_items.txt'
prompt = open(prompt_path, encoding='utf-8').read().replace('{document_text}', ans_doc['text'])
raw_resp = ingestion.llm_client.invoke(prompt)

print(f"End of raw_resp:\n{repr(raw_resp[-500:])}")

cleaned = ingestion._clean_json_str(raw_resp)
print(f"End of cleaned:\n{repr(cleaned[-500:])}")

try:
    data = json.loads(cleaned)
    print(f"JSON parsed SUCCESSFULLY! Extracted {len(data.get('items', []))} items.")
except Exception as e:
    print(f"JSON parse error: {e}")
