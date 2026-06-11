# -*- coding: utf-8 -*-
"""
Exercício 2 — Substituto discreto por EQUIVALÊNCIA À RESPOSTA AO DEGRAU
(step invariance).

Ideia: escolher Gd(z) de modo que a resposta ao degrau discreta coincida com
as AMOSTRAS da resposta ao degrau contínua:

        yd_degrau[k] = y_degrau(kh)

Como a transformada Z do degrau unitário é z/(z-1) e a transformada de
Laplace é 1/s, a condição é

        Gd(z) * z/(z-1)  =  Z{ amostras de L^{-1}[ G(s)/s ] }

        =>  Gd(z) = (1 - z^{-1}) * Z{ L^{-1}[G(s)/s] amostrada em t = kh }.

RESULTADO IMPORTANTE: o equivalente ao degrau É o equivalente ZOH.
Intuição: um degrau na entrada do conversor D/A com ZOH produz exatamente um
degrau contínuo na planta; logo, exigir que as respostas ao degrau coincidam
nas amostras é o mesmo que usar a discretização ZOH (que é exata nos
instantes de amostragem para entradas constantes por partes).
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.signal import residue
from utils import FIGS


def equivalente_degrau(num, den, h):
    """Substituto discreto por invariância ao degrau (polos de G(s)/s distintos).

    Passos:
      1. F(s) = G(s)/s = soma_i ri/(s - pi)   (inclui o polo extra em s = 0)
      2. F_amostras -> Z:  soma_i ri z/(z - e^{pi h})
      3. Gd(z) = (z-1)/z * (passo 2) = soma_i ri (z-1)/(z - e^{pi h})
    """
    den_s = np.polymul(den, [1.0, 0.0])      # multiplica por s no denominador
    r, p, _ = residue(num, den_s)
    Gd = ct.tf([0.0], [1.0], dt=h)
    for ri, pi in zip(r, p):
        if abs(pi) < 1e-9:
            # polo em s=0 (do degrau): ri (z-1)/(z-1) = ri (constante)
            Gd += ct.tf([ri.real], [1.0], dt=h)
        else:
            Gd += ct.tf([ri.real, -ri.real], [1.0, -np.exp(pi.real * h)], dt=h)
    # remove resíduos numéricos ~1e-17 do coeficiente líder do numerador
    numz = np.trim_zeros(np.where(np.abs(Gd.num[0][0]) < 1e-12, 0.0,
                                  Gd.num[0][0]), "f")
    return ct.tf(numz, Gd.den[0][0], dt=h)


def main():
    # ----- Escolhas numéricas -------------------------------------------
    # Mesma G(s) do Ex. 1 para comparar; h = 0.1 s bem visível no gráfico.
    num, den = [2.0], [1.0, 4.0, 3.0]        # G(s) = 2/((s+1)(s+3))
    h = 0.1

    G = ct.tf(num, den)
    Gd_step = equivalente_degrau(num, den, h)
    Gd_zoh = ct.sample_system(G, h, method="zoh")

    print("Gd(z) por invariância ao degrau:", Gd_step)
    print("Gd(z) por ZOH (c2d)            :", Gd_zoh)
    # Devem ser idênticas (a menos de erro numérico)
    print(f"G(0) = {ct.dcgain(G):.4f} | Gd(1) = {ct.dcgain(Gd_step):.4f} | "
          f"Gd_zoh(1) = {ct.dcgain(Gd_zoh):.4f} (ganho DC preservado)")

    Tf = 5.0
    tc, yc = ct.step_response(G, T=np.linspace(0, Tf, 1000))
    td, yd = ct.step_response(Gd_step, T=np.arange(0, Tf + h, h))
    tz, yz = ct.step_response(Gd_zoh, T=np.arange(0, Tf + h, h))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tc, yc, "b", lw=2, label="degrau contínuo $y(t)$")
    ax.plot(td, yd, "ro", ms=6, label="equiv. ao degrau (frações parciais)")
    ax.plot(tz, yz, "k+", ms=12, mew=2, label="equivalente ZOH (c2d)")
    ax.set_title(f"Equivalência à resposta ao DEGRAU = equivalente ZOH (h={h}s)\n"
                 "coincidência exata nos instantes de amostragem")
    ax.set_xlabel("t (s)"); ax.set_ylabel("y")
    ax.legend(); ax.grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex2_degrau.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
