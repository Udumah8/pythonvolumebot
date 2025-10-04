from solders.keypair import Keypair
import base58

# Input byte array (100 bytes)
byte_array = [
    128, 0, 0, 0, 0, 104, 225, 85, 110, 105, 230, 104, 171, 213, 11, 163,
    134, 26, 167, 125, 93, 118, 18, 239, 253, 84, 18, 249, 31, 237, 219, 180,
    200, 177, 5, 122, 82, 184, 220, 90, 114, 105, 122, 32, 66, 3, 174, 78, 48,
    201, 15, 254, 197, 109, 239, 130, 184, 62, 101, 212, 175, 16, 246, 242, 0,
    1, 67, 18, 177, 190, 101, 107, 141, 61, 220, 57, 1, 176, 22, 19, 10, 230,
    75, 157, 249, 28, 101, 67, 84, 147, 138, 67, 251, 173, 222, 236, 152, 222,
    247, 220, 2, 54, 49, 134, 39
]

# Take the first 32 bytes as the secret key (standard for Solana)
secret_key_bytes = byte_array[:32]

# Create a Solana Keypair from the secret key
try:
    keypair = Keypair.from_seed(secret_key_bytes)

    # Get the base58-encoded private key (full 64-byte keypair: secret + public)
    private_key_bytes = keypair.to_bytes()  # 64 bytes (32 secret + 32 public)
    private_key_base58 = base58.b58encode(private_key_bytes).decode()

    # Get the public key
    public_key = str(keypair.pubkey())

    print("Private Key (base58):", private_key_base58)
    print("Public Key:", public_key)

except Exception as e:
    print(f"Error creating keypair: {e}")
    print("The byte array may not be a valid 32-byte secret key.")