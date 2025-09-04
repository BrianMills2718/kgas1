import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import hashlib

class PiiService:
    """
    A service for encrypting and decrypting Personally Identifiable Information (PII)
    using a recoverable, key-based encryption scheme (AES-GCM).
    """

    def __init__(self, password: str, salt: bytes):
        if not password or not salt:
            raise ValueError("Password and salt must be provided for PII service.")
        self._key = self._derive_key(password, salt)

    def _derive_key(self, password: str, salt: bytes, length: int = 32) -> bytes:
        """Derives a 256-bit key from the given password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            iterations=100_000,
            backend=default_backend()
        )
        return kdf.derive(password.encode())

    def encrypt(self, plaintext: str) -> dict:
        """
        Encrypts the plaintext PII.
        
        Raises:
            TypeError: If plaintext is not a string
            ValueError: If plaintext is empty

        Returns:
            A dictionary containing:
            - pii_id: A unique, non-reversible ID for the encrypted data.
            - ciphertext: The encrypted data.
            - nonce: The unique nonce used for encryption.
        """
        # Manual validation (replaces @icontract decorators)
        if not isinstance(plaintext, str):
            raise TypeError(f"plaintext must be str, got {type(plaintext).__name__}")
        if len(plaintext) == 0:
            raise ValueError("plaintext cannot be empty")
        
        aesgcm = AESGCM(self._key)
        nonce = os.urandom(12)  # AES-GCM standard nonce size
        plaintext_bytes = plaintext.encode()
        
        ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)

        # The pii_id is a hash of the ciphertext, making it unique and non-reversible
        pii_id = hashlib.sha256(ciphertext).hexdigest()

        return {
            "pii_id": pii_id,
            "ciphertext_b64": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce_b64": base64.b64encode(nonce).decode('utf-8'),
        }

    def decrypt(self, ciphertext_b64: str, nonce_b64: str) -> str:
        """
        Decrypts the ciphertext to retrieve the original PII.
        
        Raises:
            ValueError: If inputs are empty or invalid
        """
        # Validation
        if not ciphertext_b64 or not isinstance(ciphertext_b64, str):
            raise ValueError("ciphertext_b64 must be non-empty string")
        if not nonce_b64 or not isinstance(nonce_b64, str):
            raise ValueError("nonce_b64 must be non-empty string")
        
        aesgcm = AESGCM(self._key)
        
        try:
            ciphertext = base64.b64decode(ciphertext_b64)
            nonce = base64.b64decode(nonce_b64)
            
            decrypted_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            return decrypted_bytes.decode()
        except Exception as e:
            # In a real app, log this error securely
            raise ValueError("Failed to decrypt PII. The data may be corrupt or the key incorrect.") from e

# Example Usage (for demonstration)
if __name__ == '__main__':
    # In a real application, these would come from a secure source
    # like environment variables or a secret management system.
    PII_PASSWORD = os.environ.get("KGAS_PII_PASSWORD", "a-secure-password-for-dev")
    PII_SALT = os.environ.get("KGAS_PII_SALT", "a-secure-salt-for-dev").encode()

    pii_service = PiiService(password=PII_PASSWORD, salt=PII_SALT)

    original_pii = "John Doe's phone number is 555-1234."

    encrypted_data = pii_service.encrypt(original_pii)
    print(f"Encrypted Data: {encrypted_data}")

    decrypted_pii = pii_service.decrypt(
        encrypted_data["ciphertext_b64"],
        encrypted_data["nonce_b64"]
    )
    print(f"Decrypted PII: {decrypted_pii}")

    assert original_pii == decrypted_pii
    print("\\nPII Service test successful!") 