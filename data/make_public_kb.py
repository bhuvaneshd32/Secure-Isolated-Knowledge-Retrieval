"""
Build data/public_kb.jsonl: the public knowledge base mixed into every user's
retrieval pool (paper Eq. 1).

Source: BEIR SciFact (5,183 scientific abstracts), tried in this order:
  1. the Hugging Face `datasets` library:  load_dataset("BeIR/scifact", "corpus")
  2. the zip file on the BEIR server (urllib)
  3. a small built-in sample, so everything still runs offline

Only the first --limit abstracts are kept (the project does not need all of them).

    python data/make_public_kb.py [--limit 500]
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

FALLBACK = [
    "Mitochondrial DNA is inherited maternally in most animal species.",
    "Annual check-ups are recommended and can be scheduled as medical appointments.",
    "Patients can review their medical history through the online appointment portal.",
    "Medical appointment cancellations must be made 24 hours in advance to avoid fees.",
    "Regular aerobic exercise reduces the risk of cardiovascular disease.",
    "Vitamin D deficiency is associated with reduced bone density.",
    "Antibiotics are ineffective against viral infections such as the common cold.",
    "Hypertension is a major risk factor for stroke and kidney disease.",
    "Insulin regulates blood glucose by promoting cellular glucose uptake.",
    "Vaccination programmes have eradicated smallpox worldwide.",
    "Type 2 diabetes is managed with diet, exercise and medications such as metformin.",
    "HbA1c reflects average blood glucose over the previous two to three months.",
    "Metformin is a first-line medication for type 2 diabetes.",
    "Statins such as atorvastatin lower LDL cholesterol and reduce cardiovascular events.",
    "Amlodipine is a calcium channel blocker used to treat high blood pressure.",
    "Salbutamol is a short-acting bronchodilator used for asthma symptom relief.",
    "Levothyroxine replaces thyroid hormone in patients with hypothyroidism.",
    "TSH is the primary screening test for thyroid function.",
    "Sumatriptan is used to treat acute migraine attacks.",
    "Proton pump inhibitors such as omeprazole reduce stomach acid production.",
    "SSRIs such as sertraline are commonly prescribed for depression and anxiety disorders.",
    "Ibuprofen is a nonsteroidal anti-inflammatory drug used for pain and inflammation.",
    "Cetirizine is an antihistamine used to relieve allergic rhinitis symptoms.",
    "Iron deficiency anaemia is treated with oral iron supplements such as ferrous sulfate.",
    "Amoxicillin is a penicillin-class antibiotic used for bacterial infections.",
    "Normal fasting blood glucose is generally below 100 mg/dL.",
    "A haemoglobin level below 12 g/dL in women often indicates anaemia.",
    "Elevated serum creatinine can indicate reduced kidney function.",
    "Ferritin is a blood protein that reflects the body's iron stores.",
    "Vitamin D is produced in the skin on exposure to sunlight.",
    "Normal blood pressure is considered to be below 120/80 mmHg.",
    "Osteoarthritis is a degenerative joint disease that commonly affects the knees and hips.",
    "Migraine often presents as a throbbing headache with nausea and light sensitivity.",
    "Asthma is a chronic inflammatory disease of the airways causing wheezing and breathlessness.",
    "Depression is a common mental disorder characterised by persistent low mood.",
    "Gastro-oesophageal reflux disease causes heartburn and acid regurgitation.",
    "Allergic rhinitis causes sneezing, a runny nose and itchy eyes.",
    "Health insurance claims require the claim number and the date of service.",
    "Patients should bring a list of current medications to every appointment.",
    "Prescriptions should be taken for the full prescribed duration unless advised otherwise.",
]


def _title_text(r: dict) -> str:
    title = (r.get("title") or "").strip().rstrip(".")
    return (title + ". " + (r.get("text") or "")).strip()


def _from_hf(limit: int | None) -> list[dict]:
    """BeIR/scifact corpus through the Hugging Face `datasets` library."""
    from datasets import load_dataset

    split = f"corpus[:{limit}]" if limit else "corpus"
    ds = load_dataset("BeIR/scifact", "corpus", split=split)
    return [{"chunk_id": f"scifact_{r['_id']}", "text": _title_text(r)} for r in ds]


def _from_zip(limit: int | None) -> list[dict]:
    """BeIR/scifact corpus from the zip file on the BEIR server."""
    raw = urllib.request.urlopen(SCIFACT_URL, timeout=60).read()
    docs = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith("corpus.jsonl"))
        for line in z.read(name).decode("utf-8").splitlines():
            r = json.loads(line)
            docs.append({"chunk_id": f"scifact_{r['_id']}", "text": _title_text(r)})
            if limit and len(docs) >= limit:
                break
    return docs


def build_public_kb(limit: int | None = 500, out: str = "data/public_kb.jsonl") -> int:
    docs: list[dict] = []
    for label, loader in (("Hugging Face datasets", _from_hf), ("BEIR zip download", _from_zip)):
        try:
            print(f"Loading SciFact via {label} ...")
            docs = loader(limit)
            print(f"SciFact: {len(docs)} documents")
            break
        except Exception as e:  # missing package, offline, blocked
            print(f"  {label} failed: {type(e).__name__}: {str(e)[:120]}")
    if not docs:
        print("Using the built-in fallback sample")
        docs = [{"chunk_id": f"public_{i:03d}", "text": t} for i, t in enumerate(FALLBACK)]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"Wrote {len(docs)} public chunks -> {out_path}")
    return len(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="data/public_kb.jsonl")
    a = ap.parse_args()
    build_public_kb(a.limit, a.out)


if __name__ == "__main__":
    main()
