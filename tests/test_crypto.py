from ninpy.crypto import decrypt, encrypt, generate_key


def test_crypto():
    key = generate_key()
    print(f"Key: {key.decode()}")

    message = "BOOK123panda"
    encrypted_message = encrypt(message, key)

    print(f"Encrypted Message: {encrypted_message.decode()}")
    decrypted_message = decrypt(encrypted_message, key)
    print(f"Decoded Message: {message}")

    assert message == decrypted_message
