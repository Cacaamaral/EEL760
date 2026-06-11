# -*- coding: utf-8 -*-
"""
Exercício 3(b) — Controlador PI, C(s) = Kp + Ki/s, discretizado via EULER
(forward), aplicado (i) à planta CONTÍNUA G(s)=kp/(s(s+ap)) e (ii) à versão
ZOH dessa planta.

Discretização de Euler do integrador (s* = (z-1)/h):
    C(z) = Kp + Ki * h/(z - 1)
em recorrência:
    xi(k+1) = xi(k) + h * e(k)          (integrador de Euler/forward)
    u(k)    = Kp e(k) + Ki xi(k)

Escolhas numéricas:
  - kp = ap = 1.5, h = 0.02 s (mesmos do Func.m).
  - Kp = 2.0, Ki = 0.8: o polinômio característico de malha fechada contínua
    é s^3 + ap s^2 + kp Kp s + kp Ki = s^3 + 1.5 s^2 + 3 s + 1.2.
    Routh: a1*a2 = 4.5 > a3 = 1.2  =>  estável, com margem (verificado
    numericamente abaixo). Ki moderado evita oscilação excessiva.
  - Referência: degrau unitário r = 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils import (FIGS, zoh_equiv, planta1, simula_discreta, simula_hibrida,
                   mostra_polos)


def faz_pi_euler(Kp, Ki, h, ref):
    """Cria um controlador PI discreto (Euler) com estado próprio."""
    est = {"xi": 0.0}
    def ctrl(k, t, x, y):
        e = ref(t) - y
        u = Kp * e + Ki * est["xi"]      # usa xi(k) (acumulado até e(k-1))
        est["xi"] += h * e               # xi(k+1) = xi(k) + h e(k)
        return u
    return ctrl


def main():
    kp, ap, h = 1.5, 1.5, 0.02
    Kp, Ki = 2.0, 0.8
    Tsim = 20.0
    ref = lambda t: 1.0                  # degrau unitário

    Ac, Bc, C = planta1(kp, ap)
    Phi, Gamma = zoh_equiv(Ac, Bc, h)

    # ----- Polos de malha fechada -----------------------------------------
    # Contínuo ideal: s^3 + ap s^2 + kp Kp s + kp Ki
    polos_c = np.roots([1.0, ap, kp * Kp, kp * Ki])
    mostra_polos("MF contínua (PI ideal)", polos_c)

    # Discreto: estados [x1, x2, xi];  e = r - C x
    #   x+  = Phi x + Gamma (Kp(r - Cx) + Ki xi)
    #   xi+ = xi + h (r - Cx)
    Acl = np.block([[Phi - Gamma * Kp @ C, Gamma * Ki],
                    [-h * C,               np.array([[1.0]])]])
    mostra_polos("MF discreta (PI Euler + planta ZOH)",
                 np.linalg.eigvals(Acl), discreto=True)

    # ----- (i) planta contínua + PI discreto (híbrida) --------------------
    rc = simula_hibrida(Ac, Bc, C, h, Tsim, faz_pi_euler(Kp, Ki, h, ref))
    # ----- (ii) planta ZOH + PI discreto (puramente discreta) -------------
    rd = simula_discreta(Phi, Gamma, C, h, Tsim, faz_pi_euler(Kp, Ki, h, ref))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(rc["t"], np.ones_like(rc["t"]), "k--", lw=1.5, label="referência r=1")
    ax1.plot(rc["t"], rc["y"], "b", lw=2, label="(i) planta contínua")
    ax1.plot(rd["t"][::10], rd["y"][::10], "ro", ms=4,
             label="(ii) planta ZOH (amostras)")
    ax1.set_title(f"3(b) PI via Euler ($K_p$={Kp}, $K_i$={Ki}, h={h}s) — "
                  "$G(s)=1.5/(s(s+1.5))$")
    ax1.set_ylabel("y"); ax1.legend(); ax1.grid(True)

    ax2.step(rc["tk"], rc["uk"], "b", where="post", lw=1.5,
             label="u (caso contínuo)")
    ax2.step(rd["t"], rd["u"], "r--", where="post", lw=1.2,
             label="u (caso ZOH)")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("u")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex3b_pi_euler.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
