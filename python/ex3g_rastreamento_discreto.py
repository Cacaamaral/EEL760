# -*- coding: utf-8 -*-
"""
Exercício 3(g) — Reprojeto do item (f) DIRETAMENTE para a versão ZOH da
planta (projeto no tempo discreto).

Diferença essencial em relação ao (f): em vez de aproximar o modelo interno
contínuo, constrói-se o MODELO INTERNO DISCRETO EXATO dos sinais amostrados:
  - degrau         -> polo em z = 1;
  - 10 sin(4 k h)  -> polos em z = e^{±j4h}  (sobre o círculo unitário!).
Polinômio característico do modelo interno discreto:
    (z - 1)(z^2 - 2 cos(4h) z + 1) = z^3 - (1+2c) z^2 + (1+2c) z - 1,
com c = cos(4h). Em forma companheira:
    eta(k+1) = Aim_d eta(k) + Bim_d e(k)

O sistema aumentado discreto [x; eta] usa o equivalente ZOH (Phi, Gamma) e
os polos de malha fechada são alocados em z = e^{s h} dos polos contínuos
desejados do item (f). Como os modos exógenos estão EXATAMENTE no modelo
interno (sem a distorção do Euler), o erro amostrado converge a zero.

Escolhas numéricas: h = 0.01 s e polos desejados idênticos ao (f), para a
comparação ser justa.
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from utils import (FIGS, zoh_equiv, planta2, simula_discreta, simula_hibrida,
                   mostra_polos)


def main():
    h = 0.01
    Tsim = 20.0
    ref = lambda t: 10.0 * np.sin(4.0 * t)
    dist = lambda t: 6.0 if t >= 10.0 else 0.0

    Ac, Bc, C = planta2()
    n = Ac.shape[0]
    Phi, Gamma = zoh_equiv(Ac, Bc, h)

    # ----- Modelo interno DISCRETO exato -----------------------------------
    c = np.cos(4.0 * h)
    # (z-1)(z^2 - 2c z + 1) = z^3 - (1+2c) z^2 + (1+2c) z - 1
    Aim = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, -(1.0 + 2.0 * c), (1.0 + 2.0 * c)]])
    Bim = np.array([[0.0], [0.0], [1.0]])
    mostra_polos("modelo interno discreto", np.linalg.eigvals(Aim),
                 discreto=True)  # |z|=1: z=1 e e^{±j4h} (marginal, proposital)

    # ----- Sistema aumentado discreto e alocação de polos -------------------
    Aa = np.block([[Phi, np.zeros((n, 3))],
                   [-Bim @ C, Aim]])
    Ba = np.vstack([Gamma, np.zeros((3, 1))])
    polos_s = np.array([-4.0, -5.0, -6.0, -3.0 + 4.0j, -3.0 - 4.0j])
    polos_z = np.exp(polos_s * h)                  # z = e^{s h}
    Ka = np.asarray(ct.place(Aa, Ba, polos_z))
    Kx, Keta = Ka[:, :n], Ka[:, n:]
    mostra_polos("MF discreta projetada", np.linalg.eigvals(Aa - Ba @ Ka),
                 discreto=True)

    # ----- Controlador discreto ---------------------------------------------
    def faz_ctrl():
        eta = np.zeros((3, 1))
        def ctrl(k, t, x, y):
            nonlocal eta
            e = ref(t) - y
            u = (-Kx @ x - Keta @ eta).item()
            eta = Aim @ eta + Bim * e              # modelo interno discreto exato
            return u
        return ctrl

    # ----- Simulações --------------------------------------------------------
    rd = simula_discreta(Phi, Gamma, C, h, Tsim, faz_ctrl(), dist)
    rc = simula_hibrida(Ac, Bc, C, h, Tsim, faz_ctrl(), dist)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(rc["t"], ref(rc["t"]), "k--", lw=1.2, label="r(t)=10 sin(4t)")
    axs[0].plot(rc["t"], rc["y"], "b", lw=1.4, label="planta contínua")
    axs[0].plot(rd["t"][::20], rd["y"][::20], "r.", ms=4, label="planta ZOH")
    axs[0].axvline(10, color="g", ls=":", lw=1.5, label="d = 6 entra")
    axs[0].set_title(f"3(g) Reprojeto DISCRETO do rastreamento (h={h}s)\n"
                     f"polos em z = e^{{sh}} dos polos contínuos do item (f)")
    axs[0].set_ylabel("y"); axs[0].legend(loc="lower left"); axs[0].grid(True)

    axs[1].plot(rc["t"], ref(rc["t"]) - rc["y"], "b", lw=1.2,
                label="erro (planta contínua)")
    axs[1].plot(rd["t"], ref(rd["t"]) - rd["y"], "r--", lw=1.0,
                label="erro (planta ZOH) -> 0 nas amostras")
    axs[1].axvline(10, color="g", ls=":", lw=1.5)
    axs[1].set_title("Erro de rastreamento — modelo interno discreto exato")
    axs[1].set_ylabel("e"); axs[1].legend(); axs[1].grid(True)

    axs[2].step(rc["tk"], rc["uk"], "b", where="post", lw=1.0,
                label="u (caso contínuo)")
    axs[2].axvline(10, color="g", ls=":", lw=1.5)
    axs[2].set_title("Sinal de controle")
    axs[2].set_xlabel("t (s)"); axs[2].set_ylabel("u")
    axs[2].legend(); axs[2].grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex3g_rastreamento_discreto.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
