# -*- coding: utf-8 -*-
"""
Exercício 1 — Substituto discreto por EQUIVALÊNCIA À RESPOSTA AO IMPULSO
(impulse invariance).

Ideia: escolher Gd(z) de modo que a resposta ao impulso discreta coincida
com as AMOSTRAS da resposta ao impulso contínua g(t):

        gd[k] = h * g(kh)

O fator h aparece porque o impulso discreto de Kronecker tem "área" 1·1 = 1
em um passo, enquanto o delta de Dirac tem área 1 em tempo contínuo; sem o
fator h o ganho em baixas frequências não seria preservado (Gd(1) != G(0)).

Procedimento (polos distintos): expandindo em frações parciais
        G(s) = soma_i  ri / (s - pi)    =>    g(t) = soma_i ri e^{pi t}.
Como Z{ e^{pi k h} } = z / (z - e^{pi h}), obtém-se

        Gd(z) = h * soma_i  ri * z / (z - e^{pi h}).

Cada polo contínuo pi é mapeado em z = e^{pi h} (mesmo mapeamento do ZOH),
mas os ZEROS resultam diferentes — a equivalência é da resposta ao impulso,
não da resposta ao degrau.
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.signal import residue, cont2discrete
from utils import FIGS


def equivalente_impulso(num, den, h):
    """Substituto discreto por invariância ao impulso (polos reais distintos).

    Gd(z) = h * soma_i ri z / (z - e^{pi h})
    """
    r, p, _ = residue(num, den)              # G(s) = soma ri/(s-pi)
    Gd = ct.tf([0.0], [1.0], dt=h)
    for ri, pi in zip(r, p):
        # ri/(s-pi)  ->  h ri z/(z - e^{pi h})
        Gd += ct.tf([h * ri.real, 0.0], [1.0, -np.exp(pi.real * h)], dt=h)
    return Gd


def main():
    # ----- Escolhas numéricas -------------------------------------------
    # G(s) com polos reais e distintos para a expansão em frações parciais
    # ser direta; h = 0.1 s é "grosso" de propósito, para o efeito da
    # amostragem ser visível no gráfico.
    num, den = [2.0], [1.0, 4.0, 3.0]        # G(s) = 2/((s+1)(s+3))
    h = 0.1

    G = ct.tf(num, den)
    Gd = equivalente_impulso(num, den, h)

    # Verificação independente: scipy cont2discrete(method='impulse')
    numd, dend, _ = cont2discrete((num, den), h, method="impulse")
    Gd_scipy = ct.tf(np.squeeze(numd), np.squeeze(dend), dt=h)
    print("Gd(z) via frações parciais :", Gd)
    print("Gd(z) via scipy 'impulse'  :", Gd_scipy)

    # Respostas ao impulso: a discreta, dividida por h, deve cair EXATAMENTE
    # sobre g(kh)
    Tf = 5.0
    tc, gc = ct.impulse_response(G, T=np.linspace(0, Tf, 1000))
    td, gd = ct.impulse_response(Gd, T=np.arange(0, Tf + h, h))

    # Ganho DC: Gd(1) deve aproximar G(0) (graças ao fator h)
    print(f"G(0) = {ct.dcgain(G):.4f} | Gd(1) = {ct.dcgain(Gd):.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    ax1.plot(tc, gc, "b", lw=2, label="$g(t)$ contínua")
    ax1.stem(td, gd / h, "r", markerfmt="ro", basefmt=" ",
             label="$g_d[k]/h$ (discreta)")
    ax1.set_title(f"Equivalência à resposta ao IMPULSO — $g_d[k]=h\\,g(kh)$, h={h}s")
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("amplitude")
    ax1.legend(); ax1.grid(True)

    # Contraste: a resposta ao DEGRAU do equivalente ao impulso NÃO casa
    # exatamente com a contínua (a equivalência é só do impulso)
    tc2, yc2 = ct.step_response(G, T=np.linspace(0, Tf, 1000))
    td2, yd2 = ct.step_response(Gd, T=np.arange(0, Tf + h, h))
    ax2.plot(tc2, yc2, "b", lw=2, label="degrau contínuo")
    ax2.step(td2, yd2, "r--", where="post", lw=1.5,
             label="degrau do equiv. ao impulso")
    ax2.set_title("Resposta ao degrau: equivalência ao impulso NÃO preserva o degrau")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("y")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex1_impulso.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
