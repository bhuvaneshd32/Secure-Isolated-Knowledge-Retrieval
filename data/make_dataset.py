import json
import random
from datetime import date
from pathlib import Path

from faker import Faker

DATE_START = date(2025, 1, 1)  # fixed range so the dataset is reproducible
DATE_END = date(2025, 12, 31)

DEPARTMENTS = [
    "cardiology", "dermatology", "neurology", "orthopaedics", "paediatrics",
    "oncology", "radiology", "endocrinology", "gastroenterology", "general medicine",
]  # fmt: skip

DIAGNOSES = [
    ("type 2 diabetes", "metformin"),
    ("hypertension", "amlodipine"),
    ("asthma", "salbutamol"),
    ("hypothyroidism", "levothyroxine"),
    ("migraine", "sumatriptan"),
    ("high cholesterol", "atorvastatin"),
    ("bacterial sinusitis", "amoxicillin"),
    ("acid reflux", "omeprazole"),
    ("depression", "sertraline"),
    ("osteoarthritis", "ibuprofen"),
    ("allergic rhinitis", "cetirizine"),
    ("anaemia", "ferrous sulfate"),
]  # (condition, drug)

LAB_TESTS = [
    ("HbA1c", 4.5, 9.5, "%"),
    ("fasting glucose", 70, 180, "mg/dL"),
    ("total cholesterol", 140, 280, "mg/dL"),
    ("haemoglobin", 9.0, 17.0, "g/dL"),
    ("creatinine", 0.5, 1.8, "mg/dL"),
    ("TSH", 0.3, 6.0, "mIU/L"),
    ("vitamin D", 10, 60, "ng/mL"),
    ("white cell count", 3.5, 12.0, "x10^9/L"),
    ("potassium", 3.2, 5.5, "mmol/L"),
    ("ferritin", 10, 300, "ng/mL"),
]  # (test, low, high, unit)

DOSES = ["5 mg", "10 mg", "20 mg", "50 mg", "100 mg", "250 mg", "500 mg"]
DURATIONS = ["7 days", "14 days", "30 days", "3 months", "6 months"]
RECORD_TYPES = ["appointment", "prescription", "lab", "note", "billing"]


def _date(fake):
    return fake.date_between(start_date=DATE_START, end_date=DATE_END)


def generate_synthetic_dataset(num_users=10, chunks_per_user=30, queries_per_user=3, seed=42):
    """
    Per user, chunks cycle through 5 record types (appointment, prescription,
    lab result, visit note, billing). Every chunk ends with a unique
    "Confidential reference: CANARY-<user>-<chunk>." token used to detect leaks.
    Queries are mixed-type; the first query of each user is always an
    appointment query (evaluation/selfcheck.py relies on its date/doctor).
    """
    Faker.seed(seed)
    random.seed(seed)
    fake = Faker()

    data_dir = Path(__file__).resolve().parent
    corpus_file = data_dir / "private_corpus.jsonl"
    queries_file = data_dir / "private_queries.jsonl"

    users = [f"user_{i:03d}" for i in range(num_users)]
    n_queries = 0

    with (
        open(corpus_file, "w", encoding="utf-8") as f_corp,
        open(queries_file, "w", encoding="utf-8") as f_query,
    ):
        for u_idx, user_id in enumerate(users):
            per_type = -(-chunks_per_user // len(RECORD_TYPES))  # ceil
            # sampled without replacement so each question has one clear answer
            diagnoses = random.sample(DIAGNOSES, min(per_type * 2, len(DIAGNOSES)))
            tests = random.sample(LAB_TESTS, min(per_type, len(LAB_TESTS)))
            used_doctors: set[str] = set()
            used_claims: set[str] = set()
            by_type: dict[str, list[dict]] = {t: [] for t in RECORD_TYPES}
            seen = dict.fromkeys(RECORD_TYPES, 0)
            user_chunks = []

            for c_idx in range(chunks_per_user):
                rtype = RECORD_TYPES[c_idx % len(RECORD_TYPES)]
                k = seen[rtype]
                seen[rtype] += 1
                canary = f"CANARY-{u_idx:03d}-{c_idx:03d}"
                chunk_id = f"chunk_{u_idx:03d}_{c_idx:03d}"

                doctor = fake.name()
                while doctor in used_doctors:
                    doctor = fake.name()
                used_doctors.add(doctor)
                when = _date(fake)
                extra: dict = {}

                if rtype == "appointment":
                    dept = random.choice(DEPARTMENTS)
                    body = f"Appointment on {when} with Dr. {doctor} in {dept}."
                elif rtype == "prescription":
                    cond, drug = diagnoses[k % len(diagnoses)]
                    dose, dur = random.choice(DOSES), random.choice(DURATIONS)
                    body = (
                        f"Prescription on {when} by Dr. {doctor}: {drug} {dose} daily for {dur}, "
                        f"for {cond}."
                    )
                    extra = {"condition": cond, "drug": drug}
                elif rtype == "lab":
                    test, lo, hi, unit = tests[k % len(tests)]
                    value = round(random.uniform(lo, hi), 1)
                    body = f"Lab result on {when} ordered by Dr. {doctor}: {test} = {value} {unit}."
                    extra = {"test": test, "value": value, "unit": unit}
                elif rtype == "note":
                    cond, _ = diagnoses[(per_type + k) % len(diagnoses)]
                    body = (
                        f"Visit note on {when} by Dr. {doctor}: patient reports symptoms "
                        f"consistent with {cond}. "
                        f"Follow-up advised in {random.choice(['2', '4', '8'])} weeks."
                    )
                    extra = {"condition": cond}
                else:  # billing
                    claim = f"CLM-{fake.random_number(digits=6, fix_len=True)}"
                    while claim in used_claims:
                        claim = f"CLM-{fake.random_number(digits=6, fix_len=True)}"
                    used_claims.add(claim)
                    amount = random.randrange(50, 2500, 5)
                    body = (
                        f"Billing on {when}: claim {claim} for the visit with Dr. {doctor}, "
                        f"amount due ${amount}."
                    )
                    extra = {"claim": claim, "amount": amount}

                record = {
                    "chunk_id": chunk_id,
                    "user_id": user_id,
                    "text": f"{body} Confidential reference: {canary}.",
                    "metadata": {"date": str(when), "doctor": doctor, "type": rtype, **extra},
                }
                user_chunks.append(record)
                by_type[rtype].append(record)
                f_corp.write(json.dumps(record) + "\n")

            # Mixed-type gold queries; the first is always an appointment query.
            def question(rec: dict) -> str:
                m = rec["metadata"]
                return {
                    "appointment": f"When is my appointment with Dr. {m['doctor']}?",
                    "prescription": f"What medication was I prescribed for {m.get('condition')}?",
                    "lab": f"What was my {m.get('test')} result?",
                    "note": f"What was discussed in my visit note with Dr. {m['doctor']}?",
                    "billing": f"How much was billed on claim {m.get('claim')}?",
                }[m["type"]]

            targets = [random.choice(by_type["appointment"])]
            other_types = random.sample(RECORD_TYPES[1:], queries_per_user - 1)
            targets += [random.choice(by_type[t]) for t in other_types]

            for i, target in enumerate(targets):
                c_num = int(target["chunk_id"].split("_")[-1])
                query_record = {
                    "query_id": f"q_{u_idx:03d}_{i}",
                    "user_id": user_id,
                    "question": question(target),
                    "gold_chunk_ids": [target["chunk_id"]],
                    "expected_canary": f"CANARY-{u_idx:03d}-{c_num:03d}",
                }
                f_query.write(json.dumps(query_record) + "\n")
                n_queries += 1

    print(f"Generated {num_users * chunks_per_user} chunks in {corpus_file}")
    print(f"Generated {n_queries} queries in {queries_file}")


if __name__ == "__main__":
    generate_synthetic_dataset()
    # The public knowledge base is built separately: python data/make_public_kb.py
