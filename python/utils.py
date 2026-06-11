# -*- coding: utf-8 -*-
"""
Utilidades comuns para os exercícios do Cap. 12 — Substitutos Discretos.

Notação consistente com o capítulo:
    x(kh+h) = Phi @ x(kh) + Gamma * u(kh),   y(kh) = C @ x(kh)
    Phi = e^{A h},   Gamma = int_0^h e^{A s} B ds
"""
import os
import numpy as np
from scipy.linalg import expm

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGS, exist_ok=True)


# ---------------------------------------------------------------------------
# Discretização ZOH (fórmula da matriz aumentada de Van Loan)
# ---------------------------------------------------------------------------
def zoh_equiv(A, B, h):
    """Equivalente ZOH exato: Phi = e^{Ah}, Gamma = int_0^h e^{As} B ds.

    Usa a matriz aumentada de Van Loan:
        M = [[A, B],
             [0, 0]]   =>   e^{M h} = [[Phi, Gamma],
                                       [0,   I   ]]
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * h)
    Phi = Md[:n, :n]
    Gamma = Md[:n, n:]
    return Phi, Gamma


# ---------------------------------------------------------------------------
# Plantas dos exercícios (formas canônicas controláveis)
# ---------------------------------------------------------------------------
def planta1(kp=1.5, ap=1.5):
    """G(s) = kp / (s (s + ap)).  Mesmos valores do Func.m (kp = ap = 1.5).

    Realização: x1 = posição (saída), x2 = velocidade.
        x1' = x2
        x2' = -ap x2 + kp u
    """
    Ac = np.array([[0.0, 1.0],
                   [0.0, -ap]])
    Bc = np.array([[0.0],
                   [kp]])
    C = np.array([[1.0, 0.0]])
    return Ac, Bc, C


def planta2():
    """G(s) = 3 / ((s+1)(s-5)) = 3 / (s^2 - 4s - 5)  — polo INSTÁVEL em s = 5.

    Forma canônica controlável:
        x1' = x2
        x2' = 5 x1 + 4 x2 + u
        y   = 3 x1
    """
    Ac = np.array([[0.0, 1.0],
                   [5.0, 4.0]])
    Bc = np.array([[0.0],
                   [1.0]])
    C = np.array([[3.0, 0.0]])
    return Ac, Bc, C


# ---------------------------------------------------------------------------
# Simuladores de malha fechada
# ---------------------------------------------------------------------------
def simula_discreta(Phi, Gamma, C, h, Tsim, controlador, dist=None, x0=None):
    """Malha fechada PURAMENTE DISCRETA: planta ZOH + controlador discreto.

    controlador(k, t, x, y) -> u  (gerencia seu próprio estado interno)
    dist(t) -> d (perturbação na ENTRADA da planta, amostrada em t = kh)
    """
    n = Phi.shape[0]
    x = np.zeros((n, 1)) if x0 is None else np.asarray(x0, float).reshape(n, 1)
    N = int(round(Tsim / h))
    T = np.zeros(N + 1); Y = np.zeros(N + 1)
    U = np.zeros(N + 1); D = np.zeros(N + 1)
    for k in range(N + 1):
        t = k * h
        y = (C @ x).item()
        d = 0.0 if dist is None else dist(t)
        u = controlador(k, t, x, y)
        T[k], Y[k], U[k], D[k] = t, y, u, d
        x = Phi @ x + Gamma * (u + d)          # x(kh+h) = Phi x + Gamma (u+d)
    return {"t": T, "y": Y, "u": U, "d": D}


def simula_hibrida(Ac, Bc, C, h, Tsim, controlador, dist=None, x0=None, nsub=20):
    """Malha HÍBRIDA: planta CONTÍNUA + controlador discreto com ZOH.

    Mesma estrutura do SimulaLP1DOF_ContinuousCase.m: a cada período h o
    controlador lê o estado amostrado e calcula u, que fica CONSTANTE (ZOH)
    durante o período; a planta é propagada em nsub sub-passos. A propagação
    intra-passo usa o próprio equivalente ZOH no sub-passo hc = h/nsub
    (exata, pois u e d são constantes em cada sub-passo).
    """
    n = Ac.shape[0]
    hc = h / nsub
    Phis, Gams = zoh_equiv(Ac, Bc, hc)         # propagador exato no sub-passo
    x = np.zeros((n, 1)) if x0 is None else np.asarray(x0, float).reshape(n, 1)
    N = int(round(Tsim / h))
    tc, yc, uc, dc = [], [], [], []
    tk, yk, uk = [], [], []
    for k in range(N + 1):
        t = k * h
        y = (C @ x).item()
        u = controlador(k, t, x, y)            # amostragem em t = kh
        tk.append(t); yk.append(y); uk.append(u)
        for i in range(nsub):                  # u constante no intervalo (ZOH)
            ts = t + i * hc
            d = 0.0 if dist is None else dist(ts)
            tc.append(ts); yc.append((C @ x).item()); uc.append(u); dc.append(d)
            x = Phis @ x + Gams * (u + d)
    return {"t": np.array(tc), "y": np.array(yc), "u": np.array(uc),
            "d": np.array(dc), "tk": np.array(tk), "yk": np.array(yk),
            "uk": np.array(uk)}


def mostra_polos(nome, autovalores, discreto=False):
    """Imprime autovalores e veredito de estabilidade."""
    autovalores = np.atleast_1d(autovalores)
    if discreto:
        margem = np.max(np.abs(autovalores)) - 1.0
        crit = "|z| < 1"
        extra = "  módulos: " + np.array2string(np.abs(autovalores), precision=4)
    else:
        margem = np.max(np.real(autovalores))
        crit = "Re(s) < 0"
        extra = ""
    if margem < -1e-9:
        veredito = "ESTÁVEL"
    elif margem < 1e-9:
        veredito = "MARGINALMENTE ESTÁVEL (polo na fronteira)"
    else:
        veredito = "INSTÁVEL"
    print(f"[{nome}] polos: {np.array2string(autovalores, precision=4)}{extra}"
          f"  ->  {veredito} ({crit})")
