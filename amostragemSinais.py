"""
/***************************************************
DISCRETIZATION OF AN ANALOG SIGN
 ***************************************************
This code is intended to discretize an analog signal formed by three main frequency components.
Thus, this program seeks to show the importance of choosing the appropriate sampling frequency during discretization.
In addition to understanding the importance of Nyquist's theorem,
it is clear that the wrong choice of this parameter (Fs) can lead to the reconstruction of a signal unrelated to
the original signal.

 Created 05/02/2020
 By [Izaias ALves](https://github.com/JRizaias)
 ****************************************************/
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

sin = np.sin
pi = np.pi
Tfinal=0.1
Fs=1500
Ts=1/Fs
passo=(10**(-6))

t = np.arange(0, 0.1, passo)         #t[inicio, fim, passo]
v = np.arange(0, 0.1, Ts)
txs = np.ones(len(v))                #alocaçao de espaço para vetor de tempo discreto
yxs = np.ones(len(v))                #alocaçao de espaço para vetor de tempo discreto
x =sin(50*2*pi*t)+sin(100*2*pi*t)+sin(500*2*pi*t) #Sinal a ser analisado
ind=0
#------------------Discretizaçao do sinal com frequencia de amostragem Fs---------------------------------
for i in v:
    txs[ind]=v[ind]
    yxs[ind]=sin(50*2*pi*v[ind])+sin(100*2*pi*v[ind])+sin(500*2*pi*v[ind])
    ind+=1

#------------------Trasformada de Fourier---------------------------------
y = sp.fft.fft(yxs)           # Transformada rápida de Fourier
fs = sp.fft.fftfreq(len(v), Ts) # Eixo de frequências entre -fs/2 a fs/2

#------------------Geraçao de graficos--------------------------
plt.figure(1)
plt.subplot(3, 1, 1)
plt.title('Analise de sinais com $Fs=1.5KHz $  ')
plt.plot(t,x,color='blue',label='Sinal continuo')
plt.ylabel('Amplitude')
plt.xlabel('Tempo(s)')
plt.xlim(0, 0.1)
plt.ylim()
plt.legend(loc=1)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot( txs,yxs,'o',color='red',label='Sinal amostrado')
plt.plot( t,x,color='blue',label='Sinal continuo')
plt.ylabel('Amplitude')
plt.xlabel('Tempo(s)')
plt.xlim(0, 0.1)
plt.ylim()
plt.legend(loc=1)
plt.grid()

plt.subplot(3, 1, 3)
plt.step(txs,yxs,color='black', where='post', label='Sinal reconstruido')
plt.plot(txs,yxs, 'C2o', alpha=0.5,color='red',label='Sinal amostrado' )
plt.plot(t,x,color='blue',label='Sinal continuo')
plt.ylabel('Amplitude')
plt.xlabel('Tempo(s)')
plt.xlim(0, 0.1)
plt.ylim()
plt.legend(loc=1)
plt.grid()
plt.show()

plt.figure('Figura 2')
plt.title('Espectro de frequências')
plt.grid()
plt.plot(fs, abs(y), color='blue')   # Espectro de frequências
plt.grid()
plt.show()
