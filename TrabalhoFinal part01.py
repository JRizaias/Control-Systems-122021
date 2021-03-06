#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is an adaptation of the algorithms provided by Professor Lucas Silva de Oliveira
('Controller_TF.py' and 'Tank_NL.py') and it simulates the open loop dynamics of a coupled tank system
with measurement noise and an inserted non-linearity (Gaussian), using the 'Python Control Systems' library.

This code also simulates the control loop with a first-order controller designed via the root location for the
system in question.

The main commands used in the Python Control Systems library are:
- NonlinearIOSystem (class)
- InterconnectedSystem (class)
- input_output_response (class)

-tf2io (function)
-input_output_response (function)

@authors: Izaías Alves, Lucas Silva Rodriges
"""


"""
Esse código é uma adaptação dos algoritmos fornecidos pelo professor Lucas Silva de Oliveira
('Controlador_TF.py' e 'Tanque_NL.py') e o mesmo simula a dinâmica em malha aberta de um sistema de tanques acoplados
com ruído de medição e uma não linearidade inserida (gaussiana), através do uso da biblioteca  'Python Control Systems'.

Este código ainda simula a malha de controle com um controlador de primeira ordem projetado via Lugar das raízes para o
sistema em questão.

Os principais comandos utilizados da biblioteca Python Control Systems são:
- NonlinearIOSystem (classe)
- InterconnectedSystem (classe)
- input_output_response (classe)

-tf2io (função)
-input_output_response (função)

@authors: Izaías Alves, Lucas Silva Rodriges
"""

import numpy as np               # Importando a biblioteca numpy - para ambiente matemático com vetores
import matplotlib.pyplot as plt  # Importando a biblioteca matplotlib - para plotagem
import control as clt            # Importando a biblioteca control - para a simulação da função de transferência do model

plt.close('all')   # Fechando todas as janelas com figuras

'----------------------------------------------------------------------------------------------------------------------'
#Definição do tempo de simulação e amostragem:

T = 1                     # Período de amostragem (segundos)
tf = 3000                 # Duração do teste (segundos)
t = np.arange(0, tf+T, T) # Vetor de tempo do experimento

'----------------------------------------------------------------------------------------------------------------------'

"""
A variável 'caso', define em qual situação o sistema será simulado:
caso = 0 : Simula o sistema em malha aberta com ruído.

caso = 1 : Simula o sistema em malha aberta com ruído, porém com amplitude diferente.

caso = 2 : Simula o sistema em malha aberta com ruído e perturbação (aumentou da vazão de saída 'q_out' em 5%).
"""
caso = 2   # Tipo de simulação do sistema

#Sequência de ruídos de medição da saída do sistema:

np.random.seed(55555555) # Define a propragação do gerador de números aleatórios da função random da biblioteca numpy

if caso == 0:
    np.random.seed(55555555)                # Define a propragação do gerador de números aleatórios
    s1 = np.random.normal(0, 0.03, len(t))  # Gera a sequencia de ruidos de mediçao da saida (tanque 1)
    np.random.seed(44444444)                # Define a propragação do gerador de números aleatórios
    s2 = np.random.normal(0, 0.025, len(t)) # Gera a sequencia de ruidos de mediçao da saida (tanque 2)
elif caso == 1:
    np.random.seed(105050)                  # Define a propragação do gerador de números aleatórios
    s1 = np.random.normal(0, 0.03, len(t))  # Gera a sequencia de ruidos de mediçao da saida (tanque 1)
    np.random.seed(505010)                  # Define a propragação do gerador de números aleatórios
    s2 = np.random.normal(0, 0.025, len(t)) # Gera a sequencia de ruidos de mediçao da saida (tanque 2)
else:
    np.random.seed(1000)                    # Define a propragação do gerador de números aleatórios
    s1 = np.random.normal(0, 0.03, len(t))  # Gera a sequencia de ruidos de mediçao da saida (tanque 1)
    np.random.seed(250000)                  # Define a propragação do gerador de números aleatórios
    s2 = np.random.normal(0, 0.025, len(t)) # Gera a sequencia de ruidos de mediçao da saida (tanque 2)

'----------------------------------------------------------------------------------------------------------------------'
# Função da equação diferencial do sistema que retorna os deltas das alturas dos níveis de água dos tanques 1 e 2:

def tanque_NL_update(t,x,u, params):
    """
    Essa função simula a dinâmica do sistema de tanques acoplados e a mesma retorna os deltas das alturas dos níveis de
    água dos tanques 1 e 2.
    :param t: Vetor de tempo
    :param x: Estados do sistema
    :param u: Entradas do sistema
    :param params: parametros do sistema
    :return: dh1 - Delta de altura do tanque 1, dh2 - Delta de altura do tanque 2
    """

    global caso # Importando a variável global 'caso' para definir se o sistema terá ou não perturbação

    # Verificando qual valor será atribuido a variável 'ds', esta representa a perturbação inserida no sistema
    if caso == 0:
        ds = 1
    elif caso == 1:
        ds = 1
    else:
        ds = 1.025

    h1 = x[0]  # Altura do tanque 1
    h2 = x[1]  # Altura do tanque 2
    us = u[0]  # Sinal de controle que entra no sistema não linear (entrada)
    s1 = u[1]  # Sinal de ruído (seq. 1) (entrada)
    s2 = u[2]  # Sinal de ruído (seq. 2) (entrada)

    # Parâmetos do sistema:
    rd = params.get('rd', 31)     # Raio do tanque (m)
    mu = params.get('mu', 0)      # Constante do sólido não linear
    sig = params.get('sig', 0.25) # Constante do sólido não linear

    A1 = (3 * rd / 5) * 2.7 * rd - (3 * rd / 5) * (1 / (sig * np.sqrt(2 * np.pi))) * np.cos(2.5 * np.pi * (h1 - mu)) \
    * np.exp(-((h1 - mu) ** 2) / 2 * sig ** 2) # Área da não linearidade (gaussiana)

    qin = 19 * us + 265.25                     # Vazão de entrada no instante de tempo i
    q21 = 37.4 * (h2 - h1) + 290.6             # Vazão entre os tanques 1 e 2 no instante i
    qout = 8.8 * h1 + 556                      # Vazão de saída do tanque 1

    dh1 = ((q21 - ds * qout) / A1)             # Delta da altura do tanque 1
    dh2 = ((qin - q21) / 3019)                 # Delta da altura do tanque 2

    dh1 = dh1 + s1                             # Delta 1 acrescido do ruído de medição (saida)
    dh2 = dh2 + s2                             # Delta 2 acrescido do ruido de medição (saida)
    return dh1, dh2

'----------------------------------------------------------------------------------------------------------------------'
# Instanciando o objeto ('tanque') do tipo 'NonlinearIOSystem' (classe), que é uma representação em espaço de estados
# de um sistema não linear:

tanque = clt.NonlinearIOSystem(tanque_NL_update, name='tanque', inputs = ('us', 's1', 's2'), outputs=('y1', 'y2'),
                              states=('h1', 'h2'))

print(f'------------Representação do sistema em espaço de estados------------\n{tanque}\n')

'----------------------------------------------------------------------------------------------------------------------'
# Função de trasferência do Controlador, usando a classe 'TransferFunction':

G_controller = clt.TransferFunction([1, 0.008],[1, 0.2, 0])

print(f'---------------Função de transferência do controlador----------------\nG_controller = {G_controller}\n')

'----------------------------------------------------------------------------------------------------------------------'
# Convertendo a função de transferência do controlador para um sistema do tipo I/O (entrada-saída).

controlador = clt.tf2io(G_controller, name='controlador', inputs='ys', outputs='uc')

'----------------------------------------------------------------------------------------------------------------------'
# Construindo a malha de controle por meio da classe 'InterconnectedSystem', que realiza as interconexões dos
# subsistemas (blocos) da malha e retorna um objeto do tipo sistema I/O (entrada-saída)

malha_controle = clt.InterconnectedSystem(
    (controlador, tanque), name='malha_controle',
    connections = (
        ('controlador.ys','-tanque.y1'),
        ('tanque.us','controlador.uc')),
    inplist = ('controlador.ys', 'tanque.us','tanque.s1', 'tanque.s2'),
    inputs = ('ref','u0', 's1', 's2'),
    outlist = ('tanque.y1','tanque.y2','tanque.us'),
    outputs = ('y1','y2','u'))

'----------------------------------------------------------------------------------------------------------------------'
# Definindo dos parâmetros de simulação:

amp = 27.808              # Amplitude do sinal de controle que leva o sistema para o ponto de operação (P.O)
u0 = amp*np.ones(len(t))  # Vetor do sinal de controle

#Sequência de degraus da referência (ref) aplicada ao sistema:
ref = 27*np.ones(len(t))  # Vetor da referência do sistema (Altura do nível de água do tanque 1)
ref[500:1000] = 31        # Degrau positivo de +4cm em relação ao ponto de operação
ref[1000:1500] = 27       # Degrau negativo de -4cm levando o sistema de volta para o ponto de operação
ref[1500:2000] = 23       # Degrau negativo de -4cm em relação ao ponto de operação
ref[2000:2500] = 29       # Degrau intermediário com amplitude de 6cm
ref[2500:3001] = 27       # Degrau negativo de -2cm levando o sistema de volta para o ponto de operação

h1 = 27                                          # Altura inicial do tanque 1
h2 = ((8.77 + 37.39) * h1 + 556 - 290.58)/37.39  # Altura inicial do tanque 2
X0 = [h1, h2]                                    # Vetor de estados dos tanques 1 e 2

'----------------------------------------------------------------------------------------------------------------------'
#Simulação do sistema em malha fechada com controlador implementado:
t_out, y_out, x_out = clt.input_output_response(malha_controle, t, U=[ref, u0, s1, s2], X0=[0, 0, X0[0], X0[1]],
                                               return_x=True)

'----------------------------------------------------------------------------------------------------------------------'
#Plotagem dos gráficos:

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(t_out, y_out[0, :], "b", label="h1(t)")                 # Plotando a saída do sistema (tanque 1)
plt.plot(t_out, ref, 'k--', color='black', label='Referência')   # Plotando o sinal de referência
#plt.plot(t_out, (ref + ref * 0.02), '--r',label='Margem de 2%') # Plotando a margem sup. de 2% em relação a referência
#plt.plot(t_out, (ref - ref * 0.02), '--r')                      # Plotando a margem inf. de 2% em relação a referência
plt.ylabel('h1(t)[cm]')                                          # Nomeando o eixo vertical
plt.xlabel('Times$(s)$')                                         # Nomeando o eixo horizontal
plt.xlim(0, tf)                                                  # Definindo os limites do tempo de simulação
plt.ylim(20, 35)                                                 # Definindo os limites de altura exibidos no gráfico
plt.legend(loc='lower right')                                    # Posicionando a legenda no gráfico
plt.title('Resposta temporal do tanque em malha fechada com controlador')  # Definindo o título do gráfico
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_out, y_out[1, :], color='blue', label="h2(t)")   # Plotando a saída do sistema (tanque 2)
plt.ylabel('h1(t)[cm]')                                     # Nomeando o eixo vertical
plt.xlim(0, tf)                                             # Definindo os limites do tempo de simulação
plt.ylim(30, 60)                                            # Definindo os limites de altura exibidos no gráfico
plt.legend()                                                # Definindo a legenda
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_out, y_out[2, :],'b',label='u0')               # Plotando o sinal de controle
plt.ylabel('u(t)[%]')                                     # Nomeando o eixo vertical
plt.xlabel('Times$(s)$')                                  # Nomeando o eixo horizontal
plt.xlim(0, tf)                                           # Definindo os limites do tempo de simulação
plt.ylim(0, 100)                                          # Definindo os limites de altura exibidos no gráfico
plt.legend()                                              # Definindo a legenda
plt.grid()
plt.show()