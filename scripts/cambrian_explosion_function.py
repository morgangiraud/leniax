import os
import math
import numpy as np
import matplotlib.pyplot as plt

cfd = os.path.dirname(os.path.realpath(__file__))


def f(x: float, expo: int = 2, threshold: int = 10, nb_era=10, nb_assets_at_init=1):
    if x <= 0:
        return 1

    all_th = [1]
    for i in range(1, nb_era):
        all_th.append(all_th[-1] + expo**i)
    all_th_np = np.array(all_th) * threshold

    for i, th in enumerate(all_th_np):
        if x <= th:
            y = math.log(x + 1 - threshold * (expo**i - 1)) / math.log(1 + threshold * expo**i)

            if i == 0:
                return (th - nb_assets_at_init) * y + nb_assets_at_init
            else:
                return threshold * y + i * threshold

    return 100


xs = np.arange(0, 10250)
ys = np.vectorize(f)(xs)

fig = plt.figure()

ax = fig.add_subplot(2, 1, 1)
ax.set_title("linear explosion")

log_ax = fig.add_subplot(2, 1, 2)
log_ax.set_xscale('log')
log_ax.set_title("log explosion")

line, = ax.plot(xs, ys, color='blue', lw=2)
log_line, = log_ax.plot(xs, ys, color='blue', lw=2)

fig.tight_layout()

plt.savefig(f"{cfd}/cambrian_explosion_function.png")
