# -*- coding: utf-8 -*-
"""
Exercício 3(d) — Controlador DISCRETO por realimentação de estados projetado
DIRETAMENTE sobre a versão ZOH de G(s) = 3/((s+1)(s-5)) — polo instável em
s = +5.

Projeto: discretiza-se a planta (Phi, Gamma) e aloca-se os polos DISCRETOS
de Phi - Gamma K dentro do círculo unitário com control.place.

Escolhas numéricas:
  - h = 0.05 s: o polo instável vira e^{5h} = e^{0.25} ≈ 1.284; h precisa
    ser pequeno o bastante para o controle "alcançar" a instabilidade
    (regra prática: h << 1/|p_instável| = 0.2 s).
  - Polos discretos desejados z = e^{s h} com s = {-4, -6}:
    z = {e^{-0.2}, e^{-0.3}} ≈ {0.8187, 0.7408} — resposta com constante de
    tempo ~0.2 s, sem exigir esforço de controle absurdo.
  - Regulação a zero a partir de x0 = [0.5, 0] (sem referência).
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from utils import (FIGS, zoh_equiv, planta2, simula_discreta, simula_hibrida,
                   mostra_polos)


def main():
    h = 0.05
    Tsim = 3.0
    x0 = [0.5, 0.0]

    Ac, Bc, C = planta2()
    mostra_polos("planta contínua (aberta)", np.linalg.eigvals(Ac))

    # ----- Equivalente ZOH e projeto discreto ------------------------------
    Phi, Gamma = zoh_equiv(Ac, Bc, h)
    mostra_polos("planta ZOH (aberta)", np.linalg.eigvals(Phi), discreto=True)

    polos_z = np.exp(np.array([-4.0, -6.0]) * h)   # mapeamento z = e^{sh}
    K = np.asarray(ct.place(Phi, Gamma, polos_z))
    print(f"h = {h}s | polos discretos desejados = {polos_z} | K = {K}")
    mostra_polos("MF discreta Phi - Gamma K",
                 np.linalg.eigvals(Phi - Gamma @ K), discreto=True)

    # Lei de controle puramente estática: u(k) = -K x(kh)
    ctrl = lambda k, t, x, y: (-K @ x).item()

    # ----- Simulações -------------------------------------------------------
    rd = simula_discreta(Phi, Gamma, C, h, Tsim, ctrl, x0=x0)
    rc = simula_hibrida(Ac, Bc, C, h, Tsim, ctrl, x0=x0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(rc["t"], rc["y"], "b", lw=2, label="planta contínua + controle discreto")
    ax1.plot(rd["t"], rd["y"], "ro", ms=4, label="planta ZOH (amostras)")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_title("3(d) Estabilização discreta de $G(s)=3/((s+1)(s-5))$ — "
                  f"polo instável em s=5, h={h}s\n"
                  f"polos de MF em z = {np.round(polos_z, 4)}")
    ax1.set_ylabel("y"); ax1.legend(); ax1.grid(True)

    ax2.step(rd["t"], rd["u"], "r", where="post", lw=1.5, label="u(k) (ZOH)")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("u")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex3d_estab_discreta.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
