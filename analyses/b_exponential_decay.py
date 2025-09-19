import numpy as np
import matplotlib.pyplot as plt

def get_c_exponential_decay(i, rate=0.3, q1=9, q2=1):
    return q1 * np.exp(-rate * i) + q2

T = 30

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, T + 1), [get_c_exponential_decay(i, rate=0.12) for i in range(T)], label="$b=0.12$", color="blue", marker="o")
plt.plot(np.arange(1, T + 1), [get_c_exponential_decay(i, rate=0.10) for i in range(T)], label="$b=0.10$", color="red", marker="^")
plt.plot(np.arange(1, T + 1), [get_c_exponential_decay(i, rate=0.08) for i in range(T)], label="$b=0.08$", color="orange", marker="s")
plt.plot(np.arange(1, T + 1), [get_c_exponential_decay(i, rate=0.06) for i in range(T)], label="$b=0.06$", color="green", marker="d")
plt.plot(np.arange(1, T + 1), [get_c_exponential_decay(i, rate=0.04) for i in range(T)], label="$b=0.04$", color="deeppink", marker="*")
plt.title("Exponential Decay of $c$ with Varying Rate $b$")
plt.xlabel("Experiment")
plt.ylabel("$c$")
plt.xticks(list(np.arange(1, T, 5)) + [T])
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.savefig("b_exponential_decay.pdf", transparent=True, bbox_inches="tight")
plt.show()