# -*- coding: utf-8 -*-
"""
Exercício 3(c) — Controlador CONTÍNUO por realimentação de estados com AÇÃO
INTEGRAL (regulação robusta / princípio do modelo interno) para
G(s) = kp/(s(s+ap)); implementação DISCRETA aplicada à planta contínua e à
versão ZOH.

Projeto (contínuo): sistema aumentado com o integrador do erro
        xi' = r - y
        xa  = [x; xi],  Aa = [[Ac, 0], [-C, 0]],  Ba = [Bc; 0]
        u   = -Ka xa  =  -K x - k_xi xi
O integrador no laço garante, de forma ROBUSTA (independe de conhecer kp,
ap ou d), erro nulo em regime para referência e perturbação em degrau.

Implementação discreta (Euler no integrador, ganhos contínuos mantidos):
        xi(k+1) = xi(k) + h (r - y(k));   u(k) = -K x(k) - k_xi xi(k)

Escolhas numéricas:
  - kp = ap = 1.5, h = 0.02 s.
  - Polos desejados {-2, -2.5, -3}: ~3-5x mais rápidos que o polo natural
    -ap = -1.5, e bem mais lentos que a frequência de amostragem
    (omega_s = 2*pi/h ≈ 314 rad/s) — a implementação discreta dos ganhos
    contínuos quase não degrada o projeto.
  - Perturbação na entrada d = 0.5 a partir de t = 10 s (testa a robustez).
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from utils import (FIGS, zoh_equiv, planta1, simula_discreta, simula_hibrida,
                   mostra_polos)


def faz_ctrl_int(K, kxi, h, ref):
    """Realimentação de estados + integrador de Euler (implementação discreta)."""
    est = {"xi": 0.0}
    def ctrl(k, t, x, y):
        u = (-K @ x).item() - kxi * est["xi"]
        est["xi"] += h * (ref(t) - y)        # xi(k+1) = xi(k) + h e(k)
        return u
    return ctrl


def main():
    kp, ap, h = 1.5, 1.5, 0.02
    Tsim = 20.0
    ref = lambda t: 1.0
    dist = lambda t: 0.5 if t >= 10.0 else 0.0   # degrau de perturbação

    Ac, Bc, C = planta1(kp, ap)
    n = Ac.shape[0]

    # ----- Projeto contínuo (sistema aumentado) ---------------------------
    Aa = np.block([[Ac, np.zeros((n, 1))],
                   [-C, np.zeros((1, 1))]])
    Ba = np.vstack([Bc, [[0.0]]])
    polos = [-2.0, -2.5, -3.0]
    Ka = np.asarray(ct.place(Aa, Ba, polos))
    K, kxi = Ka[:, :n], float(Ka[0, n])
    print(f"Ka = [K, k_xi] = {Ka}")
    mostra_polos("MF contínua projetada", np.linalg.eigvals(Aa - Ba @ Ka))

    # ----- Verificação da malha DISCRETA (planta ZOH + Euler no integrador)
    Phi, Gamma = zoh_equiv(Ac, Bc, h)
    Acl = np.block([[Phi - Gamma @ K, -Gamma * kxi],
                    [-h * C,          np.array([[1.0]])]])
    mostra_polos("MF discreta implementada", np.linalg.eigvals(Acl),
                 discreto=True)

    # ----- Simulações ------------------------------------------------------
    rc = simula_hibrida(Ac, Bc, C, h, Tsim, faz_ctrl_int(K, kxi, h, ref), dist)
    rd = simula_discreta(Phi, Gamma, C, h, Tsim, faz_ctrl_int(K, kxi, h, ref), dist)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(rc["t"], np.ones_like(rc["t"]), "k--", lw=1.5, label="referência r=1")
    ax1.plot(rc["t"], rc["y"], "b", lw=2, label="planta contínua")
    ax1.plot(rd["t"][::10], rd["y"][::10], "ro", ms=4, label="planta ZOH (amostras)")
    ax1.axvline(10, color="g", ls=":", lw=1.5, label="perturbação d=0.5 entra")
    ax1.set_title("3(c) Realim. de estados + ação integral (projeto contínuo, "
                  f"implem. discreta, h={h}s)\npolos desejados {polos}")
    ax1.set_ylabel("y"); ax1.legend(loc="lower right"); ax1.grid(True)

    ax2.step(rc["tk"], rc["uk"], "b", where="post", lw=1.5, label="u (caso contínuo)")
    ax2.step(rd["t"], rd["u"], "r--", where="post", lw=1.2, label="u (caso ZOH)")
    ax2.axvline(10, color="g", ls=":", lw=1.5)
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("u")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex3c_regulacao_robusta.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
