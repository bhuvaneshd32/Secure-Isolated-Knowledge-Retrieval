"""
Key-exposure experiment: what does an attacker learn from ONE leaked key?

Method A keeps a single encryption key (plus a MAC key) per user, so anyone who
obtains it can decrypt every record of that user.

Method B derives a fresh key per node through a one-way chain
(K_{i+1} = HKDF(K_i)). A key leaked from the middle of the chain opens that node
and the *later* ones, but never the earlier ones: K_{i-1} cannot be computed
from K_i. In Method B improved a swapped or altered node is also rejected.

Scenario this models: a worker, cache or log line that held the key of record j
(e.g. a half-finished batch job, a debug dump) is compromised, together with the
database file. Normal question answering gives identical results in A and B; this
is where they differ.

    python -m evaluation.key_exposure [--user user_000] [--position 20]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from privrag.store import IntegrityError, UserSession
from privrag.store.method_a import MethodAStore
from privrag.store.method_b import MethodBStore


def exposure_method_a(store: MethodAStore, session: UserSession) -> list[str]:
    """Attacker holds the user's two keys + the DB: tries every row, keeps what verifies."""
    thief = UserSession(session.user_id, {"enc": session.key("enc"), "mac": session.key("mac")})
    out = []
    for row in store.conn.execute(
        "SELECT pk, iv_emb, iv_chunk, iv_meta, c_emb, c_chunk, c_meta, tag FROM ma_records"
    ).fetchall():
        try:
            out.append(store._open_record(thief, row).text)
        except IntegrityError:
            continue
    return out


def leaked_key_at(store: MethodBStore, session: UserSession, position: int) -> bytes:
    """The key of the node at `position` (0-based) in the user's chain."""
    chain, _ = store._walk(session)
    return chain[position][1]


def exposure_method_b(store: MethodBStore, user_id: str, leaked_key: bytes) -> list[str]:
    return [c.text for c in store.read_forward_with_key(user_id, leaked_key)]


def main() -> None:
    from privrag.rag.seed import DEMO_PASSWORD

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--user", default="user_000")
    ap.add_argument("--position", type=int, default=20, help="0-based chain position of the leak")
    ap.add_argument("--demo-dir", default=str(ROOT / ".demo"))
    a = ap.parse_args()

    d = Path(a.demo_dir)
    sa = MethodAStore(str(d / "method_a.db"))
    sb = MethodBStore(str(d / "method_b_improved.db"), mode="improved")
    ses_a, ses_b = sa.unlock(a.user, DEMO_PASSWORD), sb.unlock(a.user, DEMO_PASSWORD)
    total = len(sa.load_user_data(ses_a))
    got_a = exposure_method_a(sa, ses_a)
    got_b = exposure_method_b(sb, a.user, leaked_key_at(sb, ses_b, a.position))
    print(f"{a.user} owns {total} records. One key leaks (chain position {a.position}).")
    print(f"  Method A : attacker decrypts {len(got_a):>3} of {total}")
    print(
        f"  Method B : attacker decrypts {len(got_b):>3} of {total}  "
        f"(records {a.position}..{total - 1})"
    )


if __name__ == "__main__":
    main()
