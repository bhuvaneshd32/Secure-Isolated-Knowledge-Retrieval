import pytest
from privrag.rag.ingest import TextChunker, get_text_hash

def test_chunker_respects_size():
    # Set a very small chunk size for testing
    chunker = TextChunker(chunk_size=40)
    
    text = "Short paragraph.\n\nAnother short one.\n\nA third paragraph here."
    chunks = chunker.chunk(text)
    
    # It should split them because combining them exceeds 40 characters
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 45 # Allowing a tiny margin for stripped newlines

def test_chunker_deduplication():
    chunker = TextChunker(chunk_size=40)
    
    # Text with an exact duplicate paragraph
    text = "This is a very important secret.\n\nThis is a very important secret."
    chunks = chunker.chunk(text)
    
    # The deduplication logic should catch the identical hash and only return one chunk
    assert len(chunks) == 1
    assert chunks[0] == "This is a very important secret."

def test_get_text_hash_is_deterministic():
    text = "Ensure this hashes exactly the same way every time."
    hash1 = get_text_hash(text)
    hash2 = get_text_hash(text)
    
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64 # SHA-256 hex digest length