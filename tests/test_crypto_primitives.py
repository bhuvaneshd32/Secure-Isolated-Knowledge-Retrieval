import pytest

from privrag import crypto as cp


def test_aes_cbc_round_trip_various_lengths():
    key, iv = cp.generate_key(), cp.generate_iv()
    for n in (0, 1, 15, 16, 17, 31, 32, 1000):
        m = bytes(range(256)) * 4
        m = m[:n]
        assert cp.aes_cbc_decrypt(key, iv, cp.aes_cbc_encrypt(key, iv, m)) == m


def test_aes_cbc_known_answer_nist_sp800_38a():
    # NIST SP 800-38A F.2.5 (CBC-AES256), first block. Our API adds PKCS7, so
    # compare only the first ciphertext block.
    key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
    iv = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    pt = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    expected = bytes.fromhex("f58c4c04d6e5f1ba779eabfb5f7bfbd6")
    assert cp.aes_cbc_encrypt(key, iv, pt)[:16] == expected


def test_encryption_is_randomised_by_iv():
    key = cp.generate_key()
    m = b"same plaintext"
    c1 = cp.aes_cbc_encrypt(key, cp.generate_iv(), m)
    c2 = cp.aes_cbc_encrypt(key, cp.generate_iv(), m)
    assert c1 != c2


def test_wrong_key_returns_bottom_or_garbage():
    key, other, iv = cp.generate_key(), cp.generate_key(), cp.generate_iv()
    c = cp.aes_cbc_encrypt(key, iv, b"secret message here")
    try:
        out = cp.aes_cbc_decrypt(other, iv, c)
    except cp.CryptoError:
        return
    assert out != b"secret message here"


@pytest.mark.parametrize(
    "bad",
    [
        b"",  # empty
        b"A" * 15,  # not block multiple
        b"A" * 16,  # last byte 0x41 > 16
        b"A" * 15 + b"\x00",  # k = 0
        b"A" * 14 + b"\x01\x02",  # last byte 2 but previous != 2
    ],
)
def test_unpad_rejects_invalid_padding(bad):
    with pytest.raises(cp.CryptoError):
        cp.unpad(bad)


def test_unpad_accepts_full_block_padding():
    assert cp.unpad(b"\x10" * 16) == b""
    assert cp.pad(b"") == b"\x10" * 16


def test_hmac_verify_and_tamper():
    k = cp.generate_key()
    tag = cp.hmac_sha256(k, b"iv", b"c_emb", b"c_chunk")
    assert len(tag) == cp.TAG_BYTES
    assert cp.hmac_verify(k, tag, b"iv", b"c_emb", b"c_chunk")
    assert not cp.hmac_verify(k, tag, b"iv", b"c_emb", b"c_chunk2")
    assert not cp.hmac_verify(cp.generate_key(), tag, b"iv", b"c_emb", b"c_chunk")


def test_hmac_is_unambiguous_across_field_boundaries():
    k = cp.generate_key()
    assert cp.hmac_sha256(k, b"ab", b"c") != cp.hmac_sha256(k, b"a", b"bc")


def test_hkdf_rfc5869_test_case_1():
    ikm = bytes.fromhex("0b" * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    okm = cp.hkdf(ikm, info, length=42, salt=salt)
    assert okm.hex() == (
        "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
    )


def test_scrypt_deterministic_and_salt_dependent():
    a = cp.scrypt_derive("pw", b"s" * 16, n=2**10)
    assert a == cp.scrypt_derive("pw", b"s" * 16, n=2**10)
    assert a != cp.scrypt_derive("pw", b"t" * 16, n=2**10)
    assert a != cp.scrypt_derive("pw2", b"s" * 16, n=2**10)
