import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def pwl_sigmoid(x):
    absx = np.abs(x)
    y = np.zeros_like(x, dtype=float)
    y[absx >= 5]                          = 1.0
    y[(absx >= 2.375) & (absx < 5)]      = 0.03125 * absx[(absx >= 2.375) & (absx < 5)] + 0.84375
    y[(absx >= 1)     & (absx < 2.375)]  = 0.125   * absx[(absx >= 1)     & (absx < 2.375)] + 0.625
    y[(absx >= 0)     & (absx < 1)]      = 0.25    * absx[(absx >= 0)     & (absx < 1)]     + 0.5
    y[x < 0] = 1 - y[x < 0]
    return y

x = np.linspace(-8, 8, 1000)
y_ref = sigmoid(x)
y_pwl = pwl_sigmoid(x)

mse     = np.mean((y_pwl - y_ref) ** 2)
max_err = np.max(np.abs(y_pwl - y_ref))
print(f"PWL: MSE={mse:.6f}, Max Error={max_err:.6f}")

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(x, y_ref, label='Sigmoid (ref)', linewidth=2)
ax1.plot(x, y_pwl, label='PWL', linestyle='--')
ax1.legend(); ax1.set_title('Sigmoid Approximation'); ax1.grid(True)


plt.tight_layout()
plt.savefig('sigmoid_approx.png', dpi=150)