from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

# Define constants for AES encryption
KEY = os.urandom(32)  # Securely generate a 256-bit AES key
IV = os.urandom(16)   # Securely generate a 128-bit initialization vector (IV)

# Function to encrypt the model file
def encrypt_model(model_path, encrypted_model_path):
    """
    Encrypts a model file using AES encryption.
    Args:
        model_path (str): Path to the plain model file.
        encrypted_model_path (str): Path to save the encrypted file.
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = f.read()  # Read model file

        # Encrypt the model
        encryptor = Cipher(
            algorithms.AES(KEY),
            modes.CFB(IV),
            backend=default_backend()
        ).encryptor()

        encrypted_data = encryptor.update(model_data) + encryptor.finalize()

        # Save the IV and encrypted data together
        with open(encrypted_model_path, 'wb') as f_enc:
            f_enc.write(IV + encrypted_data)

        print(f"Model encrypted and saved to {encrypted_model_path}")
    except Exception as e:
        print(f"Error during encryption: {e}")

# Main function to test the encryption process
if __name__ == "__main__":
    # Specify the paths
    original_model_path = 'complete.py'          # Original file (model or code file)
    encrypted_model_path = 'main.enc'           # Encrypted output file

    # Encrypt the model
    encrypt_model(original_model_path, encrypted_model_path)
