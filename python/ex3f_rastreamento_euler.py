# -*- coding: utf-8 -*-
"""
Exercício 3(f) — Rastreamento de r(t) = 10 sin(4t) com rejeição da
perturbação em degrau d(t) = 6·1(t-10), para G(s) = 3/((s+1)(s-5)),
projeto CONTÍNUO + discretização do controlador via EULER.

Princípio do modelo interno: o controlador deve conter os modos dos sinais
exógenos —
  - degrau (perturbação): polo em s = 0  (integrador);
  - senóide em omega = 4 (referência): polos em s = ±4j  (s^2 + 16).
Modelo interno: eta' = Aim eta + Bim e,  e = r - y, com polinômio
característico s(s^2+16) = s^3 + 16 s (forma companheira):

    Aim = [[0,1,0],[0,0,1],[0,-16,0]],  Bim = [0,0,1]^T

Sistema aumentado xa = [x; eta] (5 estados) e u = -Ka xa por alocação de
polos. Com a malha estável, o modelo interno força e(t) -> 0 mesmo com a
senóide persistente e o degrau de perturbação (regulação robusta de saída).

Discretização do CONTROLADOR via Euler:
    eta(k+1) = eta(k) + h (Aim eta(k) + Bim e(k)) = (I + h Aim) eta + h Bim e

OBS: Euler mapeia os polos ±4j do modelo interno em z = 1 ± 4jh, com
|z| = sqrt(1+16h^2) > 1 — o oscilador interno fica LEVEMENTE fora do círculo
unitário, então o rastreamento amostrado deixa de ser assintoticamente
exato (fica um pequeno erro residual; comparar com o reprojeto do item (g)).

Escolhas numéricas:
  - h = 0.01 s: pequeno frente a omega = 4 rad/s (≈157 amostras/período) e
    necessário para o Euler não degradar demais o oscilador interno.
  - Polos desejados {-4, -5, -6, -3±4j}: parte real <= -3 (mais rápida que a
    perturbação/referência) e par complexo com a "velocidade" da senóide.
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from utils import (FIGS, zoh_equiv, planta2, simula_discreta, simula_hibrida,
                   mostra_polos)


def projeto_continuo():
    """Retorna (Ac,Bc,C, Aim,Bim, Kx,Keta) do projeto contínuo aumentado."""
    Ac, Bc, C = planta2()
    n = Ac.shape[0]
    # modelo interno: s(s^2+16)
    Aim = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -16.0, 0.0]])
    Bim = np.array([[0.0], [0.0], [1.0]])
    # aumentado: [x; eta]
    Aa = np.block([[Ac, np.zeros((n, 3))],
                   [-Bim @ C, Aim]])
    Ba = np.vstack([Bc, np.zeros((3, 1))])
    polos = [-4.0, -5.0, -6.0, -3.0 + 4.0j, -3.0 - 4.0j]
    Ka = np.asarray(ct.place(Aa, Ba, polos))
    Kx, Keta = Ka[:, :n], Ka[:, n:]
    mostra_polos("MF contínua projetada", np.linalg.eigvals(Aa - Ba @ Ka))
    return Ac, Bc, C, Aim, Bim, Kx, Keta, polos


def faz_ctrl_euler(Aim, Bim, Kx, Keta, h, ref):
    """Controlador com modelo interno discretizado por Euler."""
    eta = np.zeros((3, 1))
    def ctrl(k, t, x, y):
        nonlocal eta
        e = ref(t) - y
        u = (-Kx @ x - Keta @ eta).item()
        eta = eta + h * (Aim @ eta + Bim * e)    # Euler: eta+ = (I+hAim)eta + hBim e
        return u
    return ctrl


def main():
    h = 0.01
    Tsim = 20.0
    ref = lambda t: 10.0 * np.sin(4.0 * t)
    dist = lambda t: 6.0 if t >= 10.0 else 0.0

    Ac, Bc, C, Aim, Bim, Kx, Keta, polos = projeto_continuo()
    Phi, Gamma = zoh_equiv(Ac, Bc, h)

    # ----- Autovalores da malha discreta implementada ----------------------
    # estados [x; eta]:  x+ = Phi x + Gamma u,  u = -Kx x - Keta eta
    #                    eta+ = (I + h Aim) eta - h Bim C x  (+ termos de r)
    Acl = np.block([[Phi - Gamma @ Kx, -Gamma @ Keta],
                    [-h * Bim @ C,     np.eye(3) + h * Aim]])
    mostra_polos("MF discreta (Euler + planta ZOH)", np.linalg.eigvals(Acl),
                 discreto=True)
    # módulo dos polos do modelo interno aproximado por Euler
    print(f"Euler no oscilador: |1 ± 4jh| = {abs(1 + 4j*h):.6f} (>1: levemente "
          "instável => erro residual pequeno; ver item (g))")

    # ----- Simulações -------------------------------------------------------
    rc = simula_hibrida(Ac, Bc, C, h, Tsim,
                        faz_ctrl_euler(Aim, Bim, Kx, Keta, h, ref), dist)
    rd = simula_discreta(Phi, Gamma, C, h, Tsim,
                         faz_ctrl_euler(Aim, Bim, Kx, Keta, h, ref), dist)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(rc["t"], ref(rc["t"]), "k--", lw=1.2, label="r(t)=10 sin(4t)")
    axs[0].plot(rc["t"], rc["y"], "b", lw=1.4, label="planta contínua")
    axs[0].plot(rd["t"][::20], rd["y"][::20], "r.", ms=4, label="planta ZOH")
    axs[0].axvline(10, color="g", ls=":", lw=1.5, label="d = 6 entra")
    axs[0].set_title(f"3(f) Rastreamento (modelo interno, projeto contínuo + "
                     f"Euler, h={h}s)\npolos desejados {polos}")
    axs[0].set_ylabel("y"); axs[0].legend(loc="lower left"); axs[0].grid(True)

    axs[1].plot(rc["t"], ref(rc["t"]) - rc["y"], "b", lw=1.2,
                label="erro (planta contínua)")
    axs[1].plot(rd["t"], ref(rd["t"]) - rd["y"], "r--", lw=1.0,
                label="erro (planta ZOH)")
    axs[1].axvline(10, color="g", ls=":", lw=1.5)
    axs[1].set_title("Erro de rastreamento e = r - y (note o resíduo do Euler)")
    axs[1].set_ylabel("e"); axs[1].legend(); axs[1].grid(True)

    axs[2].step(rc["tk"], rc["uk"], "b", where="post", lw=1.0,
                label="u (caso contínuo)")
    axs[2].axvline(10, color="g", ls=":", lw=1.5)
    axs[2].set_title("Sinal de controle")
    axs[2].set_xlabel("t (s)"); axs[2].set_ylabel("u")
    axs[2].legend(); axs[2].grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex3f_rastreamento_euler.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
