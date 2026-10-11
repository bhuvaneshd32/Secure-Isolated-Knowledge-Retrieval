"""
One-shot build step. Run it once before starting the UI:

    python build_kb.py                 # synthetic private corpus + public KB
    python build_kb.py --users 5 --chunks 20 --public-limit 500

It does all the slow work up front, so the UI starts instantly:

  1. generate the synthetic private corpus     -> data/private_corpus.jsonl
  2. build the public knowledge base           -> data/public_kb.jsonl
  3. embed the public KB (public, not secret)  -> data/public_kb.<embedder>.<n>.<content-hash>.npy
  4. embed every user's private chunks and encrypt them with Method A
     (embedding + chunk under AES-CBC, HMAC tag)  -> .demo/method_a.db
     Private embeddings are held in memory only; they never reach disk in clear.
  5. load the same data into Method B (paper + hardened chains)
                                                     -> .demo/method_b_<mode>.db
  6. load the same data into the plaintext baseline  -> .demo/plaintext.db
     (only so the UI can show the naive, leaky system next to A and B)
  7. write .demo/manifest.json (embedder, counts, users) for the UI
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.make_dataset import generate_synthetic_dataset
from data.make_public_kb import build_public_kb
from privrag.rag.generator import EchoLLM
from privrag.rag.ingest import get_embedder
from privrag.rag.pipeline import RagService
from privrag.rag.public_kb import load_public_kb
from privrag.rag.retriever import Retriever
from privrag.rag.seed import DEMO_PASSWORD, load_jsonl, seed_store
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore
from privrag.store.plaintext import PlaintextStore

DEMO_DIR = Path(os.getenv("PRIVRAG_DEMO_DIR", ROOT / ".demo"))
MANIFEST = DEMO_DIR / "manifest.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=30, help="private chunks per user")
    ap.add_argument("--public-limit", type=int, default=500, help="cap the public KB size")
    ap.add_argument("--embedder", default=None, help="minilm | hash (default: auto)")
    a = ap.parse_args()

    t0 = time.time()
    DEMO_DIR.mkdir(exist_ok=True)
    corpus_path = ROOT / "data" / "private_corpus.jsonl"
    public_path = ROOT / "data" / "public_kb.jsonl"

    print("[1/6] generating private corpus ...")
    generate_synthetic_dataset(num_users=a.users, chunks_per_user=a.chunks)

    print("[2/6] building public knowledge base ...")
    build_public_kb(a.public_limit, str(public_path))

    print("[3/6] embedding public knowledge base ...")
    embedder = get_embedder(a.embedder)
    for stale in (ROOT / "data").glob("public_kb.*.npy"):  # drop caches from older builds
        stale.unlink()
    pub_embs, pub_meta = load_public_kb(embedder, public_path)

    print("[4/6] embedding + encrypting private data (Method A) ...")
    for f in DEMO_DIR.glob("*.db"):
        f.unlink()
    retriever = Retriever(pub_embs, pub_meta)
    corpus = load_jsonl(corpus_path)
    enc = RagService(MethodAStore(str(DEMO_DIR / "method_a.db")), embedder, retriever, EchoLLM())
    seed_store(enc, corpus)

    print("[5/6] encrypting private data (Method B, paper + hardened chains) ...")
    for mode in ("paper", "hardened"):
        svc = RagService(
            MethodBStore(str(DEMO_DIR / f"method_b_{mode}.db"), mode=mode),
            embedder,
            retriever,
            EchoLLM(),
        )
        seed_store(svc, corpus)

    print("[6/6] loading plaintext baseline ...")
    plain = RagService(
        PlaintextStore(str(DEMO_DIR / "plaintext.db")), embedder, retriever, EchoLLM()
    )
    seed_store(plain, corpus)

    users = sorted({c["user_id"] for c in corpus})
    MANIFEST.write_text(
        json.dumps(
            {
                "embedder": getattr(embedder, "name", "minilm"),
                "public_docs": len(pub_meta),
                "private_chunks": len(corpus),
                "users": users,
                "password": DEMO_PASSWORD,
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"\nDone in {time.time() - t0:.1f}s: {len(users)} users, {len(corpus)} private chunks "
        f"(encrypted), {len(pub_meta)} public docs.\n"
        f"Start the UI with:  streamlit run demo/app.py\n"
        f"Log in as {users[0]} / {DEMO_PASSWORD}"
    )


if __name__ == "__main__":
    main()
