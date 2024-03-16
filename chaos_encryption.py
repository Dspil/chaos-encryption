from functools import reduce

# =========
# Constants
# =========

N = 8  # also the length of a row in the ncml
eps = 0.05  # epsilon
b = 0.4999  # used in the local chaotic map
N0 = 100 # constant amount of times to run NCML during initialization

AES_TABLE = [[0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
             [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
             [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
             [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
             [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
             [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
             [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
             [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
             [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
             [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
             [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
             [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
             [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
             [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],
             [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
             [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]]

# ==================
# NCML related logic
# ==================

# The tent local chaotic map
f = lambda x: x / b if x <= b else (1 - x) / (1 - b)  # (2)

# Functions for reusing bytes from the ncml
F = lambda a, b, c, d: (((a & b) | (~a & c)) + d) % 256  # (4)
G = lambda a, b, c, d: (((a & c) | (b & ~c)) + d) % 256  # (5)
H = lambda a, b, c, d: ((a ^ b ^ c) + d) % 256  # (6)
I = lambda a, b, c, d: ((b ^ (a | ~c)) + d) % 256  # (7)


def ncml(x):
    """Compute the next row of the ncml"""
    return [(1 - eps) * f(x[i]) + eps * f(x[(i + 1) % N]) for i in range(N)]


def get_ncml_bytes(x):
    """For each number in a row of the ncml, get two bytes corresponding to the 9th to 24th bits of the number after the decimal point."""
    return [
        j(i)
        for i in (int(k * 2**24) % (2**16) for k in x)
        for j in (lambda a: a // 256, lambda a: a % 256)
    ]


def generate_S_nums(A):
    """Generate numbers S0 to S15 as in (3) returning them in a list.
    Input is A0 to A15 in list A"""
    return [
        j(A[i], A[i + 1], A[i + 2], A[i + 3])
        for i in range(0, 16, 4)
        for j in (
            F,
            lambda a0, a1, a2, a3: G(a1, a2, a3, a0),
            lambda a0, a1, a2, a3: H(a2, a3, a0, a1),
            lambda a0, a1, a2, a3: I(a3, a0, a1, a2),
        )
    ]


def AES_sub(nums):
    """Substitutes all numbers in nums using the AES table"""
    return [AES_TABLE[i // 16][i % 16] for i in nums]


def gen_output_nums(Sb):
    """Generate the output numbers according to (9)"""
    return [
        ((Sb[i] ^ Sb[(i + 1) % 64]) + (Sb[(i + 2) % 64] ^ Sb[(i + 3) % 64])) % 256
        for i in range(64)
    ]


def get_64_pr_nums(x):
    """With current row of the ncml table x:
    - Get A0 to A15 (step 1) (we assumed the row of NCML x is already updated)
    - Extract S0 to S15 (step 2)
    - Rotate A0 to A15 (step 3)
    - Go to step 2 if the numbers generated are less than 64 else step 4
    - Substitute everything using the AES S-box (step 4)
    - Use (9) to compute the output pseudorandom numbers (step 5)"""
    S = []
    # Step 1
    A = get_ncml_bytes(x)

    while len(S) < 64:
        # Step 2
        S += generate_S_nums(A)
        # Step 3
        A.append(A.pop(0))

    # Step 4
    Sb = AES_sub(S)

    # Step 5
    R = gen_output_nums(Sb)
    return [[R[i * 8 + j] for j in range(8)] for i in range(8)] # reshape it as a 8x8 matrix


# ====================
# Encryption algorithm
# ====================


def cycL(x, y):
    """Cycle the bits of x to the left y times"""
    return (x % 2 ** (8 - y)) * 2 ** y + x // 2 ** (8 - y)


def LSB3(x):
    return x % 8


def encrypt(K, img):
    """ Encryption algorithm.
    For simplicity, we assume that images are 256x256 grayscale.
    img is the input image as a list of lists of values 0-255
    K is the 128 bit key
    R is the number of rounds for encryption"""

    # Step 1
    # Since we made the assumption of its size, adjusting the image is not necessary
    H = 256 # height
    W = 256 # width
    num = 256 * 256 // 64 # number of blocks
    G = 256 # number of possible grey levels

    # Find kL and r
    kL = reduce(lambda a,b: a^b, K) * num // 256 # (11)
    r = reduce(lambda a,b: a+b, K[8:]) % (H * W) # (12)

    # Step 2
    x0 = [(K[i] + 0.1) / 256 for i in range(8)] # (13) initial NCML row

    # Initialize the NCML
    x = x0
    for i in range(N0):
        x = ncml(x)

    # Step 3
    I = reduce(lambda a,b: a+b, img) # flatten image to generate the sequence I
    Iprime = [I[i+j] for j in range(r) for i in range(0, len(I) - j, r)] # permute I to generate I'
    B = [[[Iprime[i*64 + 8 * j + k] for k in range(8)] for j in range(8)] for i in range(num)] # generate the blocks of 8x8 pixels

    # Step 4 all steps are done for every block
    C = [None for k in range(num)] # init the table where the ciphered blocks will be
    Cprev = [[K[j + 8] for j in range(8)] for i in range(8)] # initialize the previous ciphered block to cover the case for B0 where the key K is used
    for k in range(num):
        Ck = [[0 for i in range(8)] for j in range(8)] # initialize the new cipher block to fill

        x = ncml(x) # iterate ncml once
        Phi = get_64_pr_nums(x) # get the next 64 pseudo random numbers

        # Step 4 (ii)
        # execute (14)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    Ck[i][j] = cycL(((B[k][i][j] ^ Phi[i][j]) + Cprev[-1][j]) % G, LSB3(Cprev[-1][(j-1) % 8] ^ Phi[i][j])) # special case for i == 0 where we have to get C[k-1]
                else:
                    Ck[i][j] = cycL(((B[k][i][j] ^ Phi[i][j]) + Ck[i-1][j]) % G, LSB3(Ck[i - 1][(j-1) % 8] ^ Phi[i][j])) # rest of the cases
        Cprev = Ck # update the previous block to be used for calculating the next Ck

        # Step 4 (iii)
        # find the new position of the block
        if k == num - 1:
            knew = kL
        else:
            knew = int(x[0] * num)
            while knew == kL or C[knew] != None:
                knew = (knew + 1) % num
        C[knew] = Ck#

        # Step 4 (iv)
        d = LSB3(Ck[7][0]) # (16)
        for i in range(4):
            if Ck[7][i] > Ck[7][(i + d) % 8]:
                x[i], x[(i + d) % 8] = x[(i + d) % 8], x[i] # exchange values in the lattice

    # Step 4 (v)
    # Notice that this step is out of the for-loop, it is done in the end and not for every block
    # in contrast to all the previous parts of step 4
    s = LSB3(kL)
    C[kL][7][7], C[0][0][s] = C[0][0][s], C[kL][7][7]

    # Get encrypted image by reshaping C
    encrypted = [[C[k+l][i][j] for l in range(4) for i in range(8) for j in range(8)] for k in range(0, num, 4)]

    return encrypted


# ====================
# Decryption algorithm
# ====================


def cycR(x, y):
    return (x % 2 ** y) * 2 ** (8 - y) + x // 2 ** y


def decrypt(K, img):
    """ Encryption algorithm.
    For simplicity, we assume that images are 256x256 grayscale.
    img is the input image as a list of lists of values 0-255
    K is the 128 bit key
    R is the number of rounds for encryption"""

    # Step 1
    # Since we made the assumption of its size, adjusting the image is not necessary
    H = 256 # height
    W = 256 # width
    num = 256 * 256 // 64 # number of blocks
    G = 256 # number of possible grey levels

    # Find kL and r
    kL = reduce(lambda a,b: a^b, K) * num // 256 # (11)
    r = reduce(lambda a,b: a+b, K[8:]) % (H * W) # (12)

    # Step 2
    x0 = [(K[i] + 0.1) / 256 for i in range(8)] # (13) initial NCML row

    # Initialize the NCML
    x = x0
    for i in range(N0):
        x = ncml(x)

    # Step 3
    IC = reduce(lambda a,b: a+b, img) # flatten image to generate the sequence I
    C = [[[IC[i*64 + 8 * j + k] for k in range(8)] for j in range(8)] for i in range(num)] # generate the blocks of 8x8 pixels

    # Step 4
    Cd = [None for k in range(num)] # init the table to figure out what knew was for each block
    Cnew = [None for k in range(num)]

    # Step 4 (i)
    s = LSB3(kL)
    C[kL][7][7], C[0][0][s] = C[0][0][s], C[kL][7][7]

    Cprev = [[K[j + 8] for j in range(8)] for i in range(8)] # initialize the previous ciphered block to cover the case for B0 where the key K is used
    for k in range(num):
    
        Ck = [[0 for i in range(8)] for j in range(8)] # initialize the new cipher block to fill

        # Step 4 (ii)
        x = ncml(x) # iterate ncml once
        Phi = get_64_pr_nums(x) # get the next 64 pseudo random numbers

        # Step 4 (iii)
        # find the original position of the block
        if k == num - 1:
            knew = kL
        else:
            knew = int(x[0] * num)
            while knew == kL or Cd[knew] != None:
                knew = (knew + 1) % num
        Cd[knew] = True
        Cnew[k] = Ck

        # execute (14)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    # special case for i == 0 where we have to get C[k-1]
                    Ck[i][j] = (Phi[i][j] ^ (cycR(C[knew][i][j], LSB3(Cprev[-1][(j - 1) % 8] ^ Phi[i][j])) - Cprev[-1][j] + G)) % G
                else:
                    # rest cases
                    Ck[i][j] = (Phi[i][j] ^ (cycR(C[knew][i][j], LSB3(C[knew][i-1][(j - 1) % 8] ^ Phi[i][j])) - C[knew][i-1][j] + G)) % G

        Cprev = C[knew] # update the previous block to be used for calculating the next Ck

        # Step 4 (v)
        d = LSB3(C[knew][7][0]) # (16)
        for i in range(4):
            if C[knew][7][i] > C[knew][7][(i + d) % 8]:
                x[i], x[(i + d) % 8] = x[(i + d) % 8], x[i] # exchange values in the lattice

    # Create Iprime from Cnew to undo step 3 of encryption
    Iprime = [j for k in Cnew for i in k for j in i]
    # Undo step 3
    I = [0 for i in Iprime]
    ind = 0
    for j in range(r):
        for i in range(0, len(Iprime) - j, r):
            I[i+j] = Iprime[ind]
            ind += 1

    # Reshape result to be an image again
    return [[I[i * 256 + j] for j in range(256)] for i in range(256)]
