import numpy as np
import matplotlib.pyplot as plt


def f_I(z, AB):
    """
    Calcule la fonction f_I(z) = (2 / pi) * arctan(2z / AB).

    Parameters:
        z (float ou array-like): Les valeurs de z.
        AB (float): La distance AB, paramètre de la fonction.

    Returns:
        float ou array-like: Les valeurs de la fonction f_I.
    """
    return (2 / np.pi) * np.arctan(2 * z / AB)


AB = [50, 100, 200]  # Distance AB (modifiable)
z_values = np.linspace(0, 500, 1000)  # Intervalle de z

plt.figure(figsize=(8, 6))

for AB in AB:
    f_values = f_I(z_values, AB)
    plt.plot(z_values, f_values, label=f"AB = {AB}")

plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
plt.xticks(np.arange(0, 500, 50))
plt.yticks(np.arange(0, 1.05, 0.1))
plt.title(
    r"Fonction $f_I(z) = \frac{2}{\pi} \tan^{-1}\left(\frac{2z}{AB}\right)$",
    fontsize=14)
plt.xlabel(r"$z$", fontsize=12)
plt.ylabel(r"$f_I(z)$", fontsize=12)
plt.legend()
plt.grid(alpha=0.5)
plt.show()
