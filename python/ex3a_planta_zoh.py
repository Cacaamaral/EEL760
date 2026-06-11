# -*- coding: utf-8 -*-
"""
Exercício 3(a) — Equivalente ZOH da planta G(s) = kp / (s (s + ap)).

Escolhas numéricas (mesmas do Func.m do repositório):
    kp = 1.5, ap = 1.5, h = 0.02 s
h = 0.02 s é rápido em relação à dinâmica da planta (constante de tempo
1/ap ≈ 0.67 s), regra prática h << constante de tempo dominante.

Phi e Gamma são calculados de TRÊS formas e comparados:
  1. Van Loan (matriz aumentada + expm)            [numérica exata]
  2. Fórmula analítica fechada                      [papel e caneta]
  3. control.sample_system(..., 'zoh')              [biblioteca]

Fórmulas analíticas (com a = ap):
    e^{At} = [[1, (1-e^{-a t})/a],
              [0,  e^{-a t}     ]]
    Phi   = e^{Ah}
    Gamma = kp * [ (h - (1-e^{-a h})/a)/a ,
                   (1-e^{-a h})/a          ]^T

A simulação aplica uma onda quadrada (constante por partes => hipótese ZOH
satisfeita) e mostra que o modelo discreto é EXATO nos instantes kh.
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from utils import FIGS, zoh_equiv, planta1, mostra_polos


def main():
    kp, ap, h = 1.5, 1.5, 0.02
    Ac, Bc, C = planta1(kp, ap)

    # ----- 1) Van Loan ---------------------------------------------------
    Phi, Gamma = zoh_equiv(Ac, Bc, h)

    # ----- 2) Fórmula analítica ------------------------------------------
    ea = np.exp(-ap * h)
    Phi_an = np.array([[1.0, (1.0 - ea) / ap],
                       [0.0, ea]])
    Gamma_an = kp * np.array([[(h - (1.0 - ea) / ap) / ap],
                              [(1.0 - ea) / ap]])

    # ----- 3) python-control ---------------------------------------------
    sysc = ct.ss(Ac, Bc, C, 0)
    sysd = ct.sample_system(sysc, h, method="zoh")

    print("Phi   (Van Loan)  =\n", Phi)
    print("Phi   (analítica) =\n", Phi_an)
    print("Phi   (c2d)       =\n", np.asarray(sysd.A))
    print("Gamma (Van Loan)  =\n", Gamma)
    print("Gamma (analítica) =\n", Gamma_an)
    print("Gamma (c2d)       =\n", np.asarray(sysd.B))
    assert np.allclose(Phi, Phi_an) and np.allclose(Phi, sysd.A)
    assert np.allclose(Gamma, Gamma_an) and np.allclose(Gamma, sysd.B)
    print(">> As três formas coincidem.\n")

    # Polos: contínuos {0, -ap} mapeados em z = e^{s h}
    mostra_polos("planta contínua", np.linalg.eigvals(Ac))
    mostra_polos("planta ZOH", np.linalg.eigvals(Phi), discreto=True)
    print(f"(verificação: e^(0*h)={np.exp(0):.4f}, e^(-ap*h)={ea:.4f})\n")

    # ----- Simulação: onda quadrada constante por partes ------------------
    # u(t) muda a cada 2 s (múltiplo de h) => o ZOH deve ser exato em t = kh
    Tsim = 8.0
    def u_de(t):
        return 1.0 if (int(t // 2) % 2 == 0) else -0.5

    # contínua "verdadeira": propagação exata em malha fina
    nfino = 20
    hc = h / nfino
    Phif, Gamf = zoh_equiv(Ac, Bc, hc)
    x = np.zeros((2, 1))
    tc, yc, uc = [], [], []
    for i in range(int(round(Tsim / hc))):
        t = i * hc
        u = u_de(h * np.floor(t / h))        # u amostrado e segurado (ZOH)
        tc.append(t); yc.append((C @ x).item()); uc.append(u)
        x = Phif @ x + Gamf * u

    # discreta ZOH
    x = np.zeros((2, 1))
    td, yd = [], []
    for k in range(int(round(Tsim / h)) + 1):
        td.append(k * h); yd.append((C @ x).item())
        x = Phi @ x + Gamma * u_de(k * h)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(tc, yc, "b", lw=2, label="planta contínua $y(t)$")
    ax1.plot(td[::10], yd[::10], "ro", ms=5,
             label="equivalente ZOH $y(kh)$ (1 a cada 10 amostras)")
    ax1.set_title(f"3(a) ZOH de $G(s)=k_p/(s(s+a_p))$, $k_p=a_p=1.5$, h={h}s\n"
                  "exato nos instantes de amostragem")
    ax1.set_ylabel("y"); ax1.legend(); ax1.grid(True)

    ax2.step(tc, uc, "g", where="post", lw=1.5, label="u(t) (constante por partes)")
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("u")
    ax2.legend(); ax2.grid(True)

    fig.tight_layout()
    fig.savefig(f"{FIGS}/ex3a_planta_zoh.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()
