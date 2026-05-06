# compare_cos.py
import numpy as np
import sys

def load_txt(path):
    return np.loadtxt(path).flatten().astype(np.float32)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

PYTHON_REF = './layer_outputs_sigmoid/sigmoid2_q.txt'
C_OUTPUT   = './sigmoid2_out_c_model_sigmoid.txt'

ref = load_txt(PYTHON_REF)
out = load_txt(C_OUTPUT)

if len(ref) != len(out):
    print(f"[ERROR] 長度不一致：python={len(ref)}, c={len(out)}")
    sys.exit(1)

# cos_sim  = cosine_similarity(ref, out)
# diff     = np.abs(ref - out)
# max_diff = diff.max()
# exact    = (diff == 0).mean() * 100
# within1  = (diff <= 1).mean() * 100

# print(f"Cosine Similarity : {cos_sim:.6f}")
# print(f"Max |diff|        : {max_diff:.0f}")
# print(f"完全一致          : {exact:.1f}%")
# print(f"±1 以內           : {within1:.1f}%")

ref = np.loadtxt(PYTHON_REF).flatten()
c   = np.loadtxt(C_OUTPUT).flatten()
diff = np.abs(ref - c)

max_diff = diff.max()
max_idx = diff.argmax()

print(f"sigmoid2 max_diff: {max_diff}, index: {max_idx}")
print(f"sigmoid2 cos_sim:  {np.dot(ref,c)/(np.linalg.norm(ref)*np.linalg.norm(c)):.6f}")