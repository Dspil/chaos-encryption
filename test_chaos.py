from random import randint as rand
from chaos_encryption import encrypt, decrypt

# 128bit key (randomly picked) split into 16 bytes
K = [rand(0, 255) for i in range(16)]

img = [
    [rand(0, 255) for i in range(256)] for j in range(256)
]  # generate a random image
enc = encrypt(K, img)
dec = decrypt(K, enc)

# assert that encrypt(decrypt(img)) == img
assert all(all(dec[i][j] == img[i][j] for j in range(256)) for i in range(256))
print("Test passed!")
