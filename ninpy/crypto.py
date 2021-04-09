"""Basic cryptography to encode or decode a string with a key.
Modified: https://stackoverflow.com/questions/2490334/simple-way-to-encode-a-string-according-to-a-password
"""
from cryptography.fernet import Fernet


def generate_key() -> bytes:
    key = Fernet.generate_key()
    return key


def encrypt(message: str, key: bytes) -> bytes:
    assert isinstance(message, str)
    # Convert to byte from string.
    message = message.encode()
    return Fernet(key).encrypt(message)


def decrypt(token: bytes, key: bytes) -> str:
    message = Fernet(key).decrypt(token)
    return message.decode()


if __name__ == '__main__':
    key = generate_key()
    print(key.decode())

    bytekey = key.encode()
    message = 'BOOK123panda'
    encrypt = encrypt(message, bytekey)

    print(f'Encrypted Message: {encrypt.decode()}')
    decrypted_message = decrypt(encrypt, key)
    print(f'Decoded Message: {message}')
    assert message == decrypted_message
