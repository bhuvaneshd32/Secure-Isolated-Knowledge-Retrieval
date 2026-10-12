from pathlib import Path

STORE_KINDS = ("plaintext", "A", "B", "B_improved")


def make_store(kind: str, db_path: str | None = None, **kwargs):
    """
    Build a store by name. Unknown / not-yet-implemented kinds fail loudly
    (no silent fallback to the insecure plaintext baseline).
    """
    if kind not in STORE_KINDS:
        raise ValueError(f"Unknown store '{kind}'. Expected one of {STORE_KINDS}")
    if db_path is None:
        Path("data").mkdir(exist_ok=True)
        db_path = f"data/privrag_{kind}.db"

    if kind == "plaintext":
        from privrag.store.plaintext import PlaintextStore

        return PlaintextStore(db_path)
    if kind == "A":
        from privrag.store.method_a import MethodAStore

        return MethodAStore(db_path, **kwargs)
    from privrag.store.method_b import MethodBStore

    return MethodBStore(db_path, mode="improved" if kind == "B_improved" else "base", **kwargs)
