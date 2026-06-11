import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

class Simulacao1DOF:
    """
    Classe para encapsular os parametros e funcoes do sistema de 1 Grau de Liberdade.
    Equivalente ao Func.m original.
    """
    def __init__(self, h=0.02, Tsimu=480):
        # Configuracoes de Simulacao
        self.h = h
        self.Tsimu = Tsimu
        self.k_max = int(Tsimu / h)
        
        # Parametros da Planta Continua G(s) = kp / (s * (s + ap))
        self.kp = 1.5
        self.ap = self.kp
        
        # Matrizes do Espaco de Estados Continuo
        self.Ac = np.array([[0, 1], [0, -self.ap]])
        self.Bc = np.array([[0], [self.kp]])
        self.Cc = np.array([[1, 0]])
        self.Dc = np.array([[0]])
        
        # Discretizacao via Segurador de Ordem Zero (ZOH)
        # cont2discrete retorna uma tupla, pegamos as 4 primeiras matrizes
        self.A, self.B, self.C, self.D, _ = signal.cont2discrete(
            (self.Ac, self.Bc, self.Cc, self.Dc), self.h, method='zoh'
        )
        
        # Projeto do Controlador (Alocacao de Polos)
        # Polos desejados em malha fechada
        polos_desejados = np.array([0.991, 0.994])
        resultado_place = signal.place_poles(self.A, self.B, polos_desejados)
        self.K = resultado_place.gain_matrix
        
        # Verificacao de estabilidade do ganho inicial
        autovalores = np.linalg.eigvals(self.A - self.B @ self.K)
        if np.max(np.abs(autovalores)) < 1.0:
            print("Status: Ganho inicial estabilizante.")
        else:
            print("Status: Ganho inicial NAO estabilizante.")
            
        # Inicializacao de Estados e Sinais
        self.x = np.array([[10.0], [0.0]]) # Condicao inicial da planta [posicao, velocidade]
        self.xi = 0.0                      # Estado do integrador discreto
        
        # Buffers para armazenamento de dados (Log)
        # Colunas: [Tempo, Saida(y), Referencia(r), Controle(u), Disturbio(d)]
        self.buff = np.zeros((self.k_max + 1, 5))

    def iterar(self):
        """
        Executa o laco de simulacao puramente discreto.
        Equivalente ao SimulaLP1DOF_DiscreteCase.m.
        """
        for k in range(self.k_max + 1):
            tempo_atual = k * self.h
            
            # 1. Leitura da Saida da Planta e Referencia
            y = self.C @ self.x
            y_val = y[0, 0]
            
            # Sinal de referencia (mantido em 0 como no script original)
            r = 0.0 * 2.0 * np.cos(1.0 * self.h * k)
            erro = y_val - r
            
            # 2. Computo do Sinal de Controle (Realimentacao de Estados)
            # u = -Kx + termo_integral
            u = - (self.K @ self.x) + 0.0001 * self.xi
            u_val = u[0, 0]
            
            # 3. Atualizacao do Disturbio
            if tempo_atual > (self.Tsimu / 3.0):
                d = 0.0 * 0.5  # Modifique aqui para testar a rejeicao a disturbios
            else:
                d = 0.0 * 0.5
                
            # 4. Armazenamento de Dados (Log)
            self.buff[k, :] = [tempo_atual, y_val, r, u_val, d]
            
            # 5. Atualizacao dos Estados para o instante k+1
            self.xi = self.xi + self.h * erro
            self.x = self.A @ self.x + self.B * (u_val + d)

    def plotar_resultados(self):
        """
        Gera os graficos de posicao e esforco de controle.
        Equivalente ao plotDOF.m.
        """
        tempo = self.buff[:, 0]
        y_saida = self.buff[:, 1]
        referencia = self.buff[:, 2]
        u_controle = self.buff[:, 3]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Grafico 1: Rastreamento de Posicao
        ax1.plot(tempo, referencia, 'k--', linewidth=2, label='Referencia (r)')
        ax1.plot(tempo, y_saida, 'b', linewidth=2, label='Posicao (y)')
        ax1.set_title('Rastreamento de Posicao para 1DOF', fontweight='bold')
        ax1.set_ylabel('Posicao p')
        ax1.legend()
        ax1.grid(True)

        # Grafico 2: Sinal de Controle
        ax2.plot(tempo, u_controle, 'b', linewidth=2, label='Esforco (u)')
        ax2.set_title('Sinal de Controle para 1DOF', fontweight='bold')
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('u (m/s)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# Execucao do Codigo
if __name__ == "__main__":
    simulacao = Simulacao1DOF(h=0.02, Tsimu=480)
    simulacao.iterar()
    simulacao.plotar_resultados()