import json
import random
from pathlib import Path
from faker import Faker

def generate_synthetic_dataset(num_users=10, chunks_per_user=30, seed=42):
    Faker.seed(seed)
    random.seed(seed)
    fake = Faker()
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    corpus_file = data_dir / "private_corpus.jsonl"
    queries_file = data_dir / "private_queries.jsonl"
    
    users = [f"user_{i:03d}" for i in range(num_users)]
    
    with open(corpus_file, "w", encoding="utf-8") as f_corp, \
         open(queries_file, "w", encoding="utf-8") as f_query:
        
        for u_idx, user_id in enumerate(users):
            user_chunks = []
            
            # 1. Generate 30 appointment chunks with unique canary tokens
            for c_idx in range(chunks_per_user):
                canary = f"CANARY-{u_idx:03d}-{c_idx:03d}"
                doctor = fake.name()
                ailment = fake.word()
                date = fake.date_this_year()
                
                # The paper's style of appointment chunk
                text = f"Appointment on {date} with Dr. {doctor} regarding {ailment}. Confidential reference: {canary}."
                chunk_id = f"chunk_{u_idx:03d}_{c_idx:03d}"
                
                record = {
                    "chunk_id": chunk_id,
                    "user_id": user_id,
                    "text": text,
                    "metadata": {"date": str(date), "doctor": doctor}
                }
                user_chunks.append(record)
                f_corp.write(json.dumps(record) + "\n")
            
            # 2. Generate 2 gold-labelled private queries per user
            # We randomly pick two chunks and ask a question that requires retrieving them
            target_chunks = random.sample(user_chunks, 2)
            for i, target in enumerate(target_chunks):
                doc_name = target["metadata"]["doctor"]
                question = f"When is my appointment with Dr. {doc_name}?"
                
                query_record = {
                    "query_id": f"q_{u_idx:03d}_{i}",
                    "user_id": user_id,
                    "question": question,
                    "gold_chunk_ids": [target["chunk_id"]],
                    "expected_canary": f"CANARY-{u_idx:03d}-{int(target['chunk_id'].split('_')[-1]):03d}"
                }
                f_query.write(json.dumps(query_record) + "\n")
                
    print(f"Generated {num_users * chunks_per_user} chunks in {corpus_file}")
    print(f"Generated {num_users * 2} queries in {queries_file}")

if __name__ == "__main__":
    generate_synthetic_dataset()
    # Note: BEIR SciFact public KB downloading logic will be added here on Day 4 for the final quality eval.