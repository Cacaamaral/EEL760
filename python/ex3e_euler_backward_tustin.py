# -*- coding: utf-8 -*-
"""
Exercício 3(e) — Regulação robusta CONTÍNUA (realim. de estados + integrador)
de G(s) = 3/((s+1)(s-5)); o CONTROLADOR é aproximado pelos três métodos do
capítulo (a única dinâmica do controlador é o integrador do erro):

    Euler    : xi(k+1) = xi(k) + h e(k)            s* = (z-1)/h
    Backward : xi(k)   = xi(k-1) + h e(k)          s* = (z-1)/(h z)
    Tustin   : xi(k)   = xi(k-1) + (h/2)(e(k)+e(k-1))   s* = (2/h)(z-1)/(z+1)

e u(k) = -K x(k) - k_xi xi(k), com os ganhos do projeto CONTÍNUO.

Escolhas numéricas:
  - Polos contínuos desejados {-3, -4, -5}: estabilizam o polo instável +5
    com dinâmica ~3x mais rápida que ele.
  - h = 0.05 s: razoavelmente grande de propósito, para as diferenças entre
    os métodos ficarem visíveis (com h -> 0 os três coincidem).
  - r = 1 (degrau) e perturbação d = 1 na entrada a partir de t = 5 s.
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from utils import (FIGS, zoh_equiv, planta2, simula_discreta, simula_hibrida,
                   mostra_polos)


def faz_ctrl(metodo, K, kxi, h, ref):
    """Realimentação de estados + integrador discretizado pelo método pedido."""
    est = {"xi": 0.0, "e_ant": 0.0}
    def ctrl(k, t, x, y):
        e = ref(t) - y
        if metodo == "euler":            # usa xi(k) ANTES de acumular e(k)
            u = (-K @ x).item() - kxi * est["xi"]
            est["xi"] += h * e
        elif metodo == "backward":       # acumula e(k) ANTES de usar
            est["xi"] += h * e
            u = (-K @ x).item() - kxi * est["xi"]
        elif metodo == "tustin":         # média trapezoidal de e(k) e e(k-1)
            est["xi"] += 0.5 * h * (e + est["e_ant"])
            est["e_ant"] = e
            u = (-K @ x).item() - kxi * est["xi"]
        return u
    return ctrl


def acl_discreta(metodo, Phi, Gamma, C, K, kxi, h):
    """Matriz de malha fechada discreta (planta ZOH + método) p/ autovalores.

    Estados: [x; xi_mem; e_mem] onde xi_mem é o acumulador armazenado e
    e_mem o erro anterior (usado só no Tustin). Termos da referência são
    ignorados (não afetam autovalores).
    """
    n = Phi.shape[0]
    Z1 = np.zeros((1, 1)); I1 = np.eye(1)
    if metodo == "euler":
        # u = -Kx - kxi xi ; xi+ = xi - hCx
        A = np.block([[Phi - Gamma @ K, -Gamma * kxi],
                      [-h * C,          I1]])
    elif metodo == "backward":
        # xi(k) = xi_mem - hCx ; u = -Kx - kxi xi(k) ; xi_mem+ = xi(k)
        A = np.block([[Phi - Gamma @ (K - kxi * h * C), -Gamma * kxi],
                      [-h * C,                          I1]])
    else:  # tustin
        # xi(k) = xi_mem + h/2 (e + e_ant), e = -Cx
        A = np.block(
            [[Phi - Gamma @ (K - kxi * 0.5 * h * C), -Gamma * kxi, -Gamma * kxi * 0.5 * h],
             [-0.5 * h * C,                          I1,           0.5 * h * I1],
             [-C,                                    Z1,           Z1]])
    return A


def main():
    h = 0.05
    Tsim = 10.0
    ref = lambda t: 1.0
    dist = lambda t: 1.0 if t >= 5.0 else 0.0

    Ac, Bc, C = planta2()
    n = Ac.shape[0]

    # ----- Projeto contínuo aumentado (como no 3c) -------------------------
    Aa = np.block([[Ac, np.zeros((n, 1))],
                   [-C, np.zeros((1, 1))]])
    Ba = np.vstack([Bc, [[0.0]]])
    polos = [-3.0, -4.0, -5.0]
    Ka = np.asarray(ct.place(Aa, Ba, polos))
    K, kxi = Ka[:, :n], float(Ka[0, n])
    print(f"Ka = [K, k_xi] = {Ka}")
    mostra_polos("MF contínua projetada", np.linalg.eigvals(Aa - Ba @ Ka))

    Phi, Gamma = zoh_equiv(Ac, Bc, h)

    metodos = ["euler", "backward", "tustin"]
    cores = {"euler": "tab:red", "backward": "tab:green", "tustin": "tab:blue"}
    res_d, res_c = {}, {}
    for m in metodos:
        mostra_polos(f"MF discreta ({m})",
                     np.linalg.eigvals(acl_discreta(m, Phi, Gamma, C, K, kxi, h)),
                     discreto=True)
        res_d[m] = simula_discreta(Phi, Gamma, C, h, Tsim,
                                   faz_ctrl(m, K, kxi, h, ref), dist)
        res_c[m] = simula_hibrida(Ac, Bc, C, h, Tsim,
                                  faz_ctrl(m, K, kxi, h, ref), dist)

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    # (1) caso discreto: planta ZOH
    axs[0].plot([0, Tsim], [1, 1], "k--", lw=1.2, label="r = 1")
    for m in metodos:
        axs[0].plot(res_d[m]["t"], res_d[m]["y"], color=cores[m], lw=1.6, label=m)
    axs[0].axvline(5, color="gray", ls=":", label="d = 1 entra")
    axs[0].set_title(f"3(e) Regulação robusta de $3/((s+1)(s-5))$ — h={h}s, "
                     f"polos {polos}\nCaso DISCRETO (planta ZOH)")
    axs[0].set_ylabel("y"); axs[0].legend(loc="lower right"); axs[0].grid(True)

    # (2) caso contínuo: planta contínua + controlador discreto
    axs[1].plot([0, Tsim], [1, 1], "k--", lw=1.2, label="r = 1")
    for m in metodos:
        axs[1].plot(res_c[m]["t"], res_c[m]["y"], color=cores[m], lw=1.6, label=m)
    axs[1].axvline(5, color="gray", ls=":")
    axs[1].set_title("Caso CONTÍNUO (planta contínua + controlador discreto)")
    axs[1].set_ylabel("y"); axs[1].legend(loc="lower right"); axs[1].grid(True)

    # (3) sinais de controle (caso contínuo)
    for m in metodos:
        axs[2].step(res_c[m]["tk"], res_c[m]["uk"], color=cores[m], lw=1.2,
                    where="post", label=m)
    axs[2].axvline(5, color="gray", ls=":")
    axs[2].set_title("Sinal de controle u(k) (caso contínuo)")
    axs[2].set_xlabel("t (s)"); axs[2].set_ylabel("u")
    axs[2].legend(); axs[2].grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex3e_metodos.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
