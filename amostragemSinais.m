% %  ***************************************************
% % DISCRETIZATION OF AN ANALOG SIGN
% %  ***************************************************
% % This code is intended to discretize an analog signal formed by three main frequency components.
% % Thus, this program seeks to show the importance of choosing the appropriate sampling frequency during discretization.
% % In addition to understanding the importance of Nyquist's theorem,
% % it is clear that the wrong choice of this parameter (Fs) can lead to the reconstruction of
%% a signal unrelated to
% % the original signal.
% % 
% %  Created 04/02/2020
% %  By [Izaias ALves](https://github.com/JRizaias)
% %  ****************************************************/

clear all
clc
Tfinal=0.1;
Fs=1500;                    %Frequencia de amostragem               
Ts=1/Fs;                    %Periodo de amostragem             
MfreqSinal=800;             %Limite superior para maior frequncia no espectro
                            %de frequencia 
passo=(10^(-6));            %Passo no vetor de tempo continuo

t=0:passo:Tfinal;           %Vetor no tempo continuo           
x =sin(50*2*pi*t)+sin(100*2*pi*t)+sin(500*2*pi*t); %Sinal a ser amostrado

i=1;
for v = 0:+Ts:Tfinal        %Amostragem de sinal
   txs(i)= v;               %Vetor no tempo discreto                                           
   yxs(i)=sin(50*2*pi*v)+sin(100*2*pi*v)+sin(500*2*pi*v);     
   i=i+1;
end

%------------------espetro de frequencia do sinal amostrado-----------------------------------
yfft = fft(yxs);     
% f = (0:length(xfft)-1)*MfreqSinal/length(xfft);
nfft = length(yxs);                         
fshift = (-nfft/2:nfft/2-1)*(2*MfreqSinal/nfft);
yshift = fftshift(yfft);

%------------------Gera�ao de graficos-----------------------------------
figure(1)
subplot(3,1,1);

plot(t,x, 'r');
title('Sinal continuo no tempo');
xlabel('time');
%------------------------
subplot(3,1,2);

%plot(n,xs, '*');
        %ou
%plot(txs,yxs, 'o');
        %ou
stem(txs,yxs);
title('Sinal discreto no tempo (Amostragem)');
xlabel('time');
%------------------------
subplot(3,1,3);
stairs(txs,yxs);
hold on 
%stem(n,xs);
plot(t,x)
title('Sinal reconstruido');
xlabel('time');

%------------------------
figure(2)
plot(fshift,abs(yshift))
title('Magnitude')
