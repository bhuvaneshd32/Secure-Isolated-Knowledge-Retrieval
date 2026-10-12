"""
Cryptographic primitives used by Method A (and later Method B).

Maps to the paper (arXiv 2503.15548):
  * AES-CBC encryption / decryption with a random IV          -> Eq. 7, 8, 14
  * PKCS#7 padding validity check, returning ⊥ on failure     -> Eq. 9
  * HMAC-SHA256 tag and verification                          -> Eq. 10, 13, 27
  * Uniform random key / IV sampling  K <-$ {0,1}^lambda      -> Eq. 5, 6

lambda (the security parameter) is fixed at 256 bits for keys and 128 bits for
the AES block / IV.

The paper's "⊥" (bottom) result is modelled by raising CryptoError. Callers
(the stores) translate it to AuthenticationError / IntegrityError.
"""

import hashlib
import hmac as _hmac
import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

KEY_BYTES = 32  # lambda = 256 bits (AES-256 / HMAC key)
IV_BYTES = 16  # AES block size
BLOCK = 16
TAG_BYTES = 32  # HMAC-SHA256 output
NONCE_BYTES = 12  # AES-GCM nonce


class CryptoError(Exception):
    """The paper's ⊥: decryption, padding or MAC verification failed."""


# --------------------------------------------------------------------------
# Randomness (Eq. 5, 6)
# --------------------------------------------------------------------------
def random_bytes(n: int) -> bytes:
    return os.urandom(n)


def generate_key() -> bytes:
    """K <-$ {0,1}^lambda"""
    return os.urandom(KEY_BYTES)


def generate_iv() -> bytes:
    """Fresh IV for every single encryption (Eq. 7)."""
    return os.urandom(IV_BYTES)


# --------------------------------------------------------------------------
# PKCS#7 padding (Eq. 9)
# --------------------------------------------------------------------------
def pad(data: bytes) -> bytes:
    k = BLOCK - (len(data) % BLOCK)  # k in [1, 16]
    return data + bytes([k]) * k


def unpad(padded: bytes) -> bytes:
    """
    Eq. 9: if there exists k in [1,16] such that the last k bytes all equal k,
    return m_pad[0 : |m_pad| - k]; otherwise return ⊥.
    """
    if not padded or len(padded) % BLOCK != 0:
        raise CryptoError("bad padded length")
    k = padded[-1]
    if k < 1 or k > BLOCK or k > len(padded):
        raise CryptoError("bad padding")
    # Check every padding byte without early exit.
    bad = 0
    for b in padded[-k:]:
        bad |= b ^ k
    if bad:
        raise CryptoError("bad padding")
    return padded[:-k]


# --------------------------------------------------------------------------
# AES-256-CBC (Eq. 7, 8, 14)
# --------------------------------------------------------------------------
def aes_cbc_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """c = AES-CBC.Enc(K, IV, m) with PKCS#7 padding."""
    _check_key_iv(key, iv)
    enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    return enc.update(pad(plaintext)) + enc.finalize()


def aes_cbc_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    """m = AES-CBC.Dec(K, IV, c); raises CryptoError (⊥) on invalid padding."""
    _check_key_iv(key, iv)
    if not ciphertext or len(ciphertext) % BLOCK != 0:
        raise CryptoError("bad ciphertext length")
    dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    padded = dec.update(ciphertext) + dec.finalize()
    return unpad(padded)


def _check_key_iv(key: bytes, iv: bytes) -> None:
    if len(key) != KEY_BYTES:
        raise CryptoError("AES-256 key must be 32 bytes")
    if len(iv) != IV_BYTES:
        raise CryptoError("IV must be 16 bytes")


# --------------------------------------------------------------------------
# HMAC-SHA256 (Eq. 10, 13, 27)
# --------------------------------------------------------------------------
def hmac_sha256(key: bytes, *parts: bytes) -> bytes:
    """
    HMAC over a *length-prefixed* concatenation of `parts`.

    The paper writes IV || c_emb || c_chunk; plain concatenation is ambiguous
    (boundary shifting), so every part is prefixed with its 8-byte length.
    """
    h = _hmac.new(key, digestmod=hashlib.sha256)
    for p in parts:
        h.update(struct.pack(">Q", len(p)))
        h.update(p)
    return h.digest()


def hmac_verify(key: bytes, tag: bytes, *parts: bytes) -> bool:
    """Constant-time comparison of the recomputed tag with the stored one."""
    return _hmac.compare_digest(hmac_sha256(key, *parts), tag)


# --------------------------------------------------------------------------
# Key derivation (used to protect the per-user keys at rest)
# --------------------------------------------------------------------------
def scrypt_derive(password: str, salt: bytes, n: int = 2**14, r: int = 8, p: int = 1) -> bytes:
    return Scrypt(salt=salt, length=KEY_BYTES, n=n, r=r, p=p).derive(password.encode("utf-8"))


def hkdf(ikm: bytes, info: bytes, length: int = KEY_BYTES, salt: bytes | None = None) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


# --------------------------------------------------------------------------
# Hash, XOR and AES-GCM (Method B: Eq. 3, 15-24; Method B improved)
# --------------------------------------------------------------------------
def sha256(*parts: bytes) -> bytes:
    """H(.) over the plain concatenation of `parts` (callers pass fixed-width parts)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def xor_bytes(a: bytes, b: bytes) -> bytes:
    if len(a) != len(b):
        raise CryptoError("xor operands must have equal length")
    return bytes(x ^ y for x, y in zip(a, b, strict=True))


def aes_gcm_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    if len(key) != KEY_BYTES or len(nonce) != NONCE_BYTES:
        raise CryptoError("bad AES-GCM key or nonce length")
    return AESGCM(key).encrypt(nonce, plaintext, aad)


def aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    """Authenticated decryption; any failure is the paper's bottom."""
    if len(key) != KEY_BYTES or len(nonce) != NONCE_BYTES:
        raise CryptoError("bad AES-GCM key or nonce length")
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag:
        raise CryptoError("AES-GCM authentication failed") from None
