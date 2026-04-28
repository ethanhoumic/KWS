import numpy as np

S_lut = 16 / 255
Z_lut = 128

with open("sigmoid_lut.txt", "w") as f:
    for i in range(256):
        x = S_lut * (i - Z_lut)
        y = 1.0 / (1.0 + np.exp(-x))
        f.write(f"{int(round(y * 255))}\n")