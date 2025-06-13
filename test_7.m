%% ��7.1Harr�˲������߶Ⱥ�С������
clc,clear,close all;
[Lo_D,Hi_d,Lo_R,hi_R]=wfilters('haar')
waveinfo('haar');
[phi,psi,xval]=wavefun('haar',10);
xaxis = zeros(size(xval));
subplot(121);plot(xval,phi,'k',xval,xaxis,'--k');
axis([0 1 -1.5 1.5]);axis square;
title('Haar�߶Ⱥ���');
subplot(122);plot(xval,psi,'k',xval,xaxis,'--k');
axis([0 1 -1.5 1.5]);axis square;
title('HarrС������');
%% ��7.2ʹ��Haar�˲����ļ򵥿���С���任��FWT��
clc,clear,close all;
f=magic(4)%ħ���������жԽ���Ԫ��֮�����
[c1,s1]=wavedec2(f,1,'haar')
[c2,s2]=wavedec2(f,2,'haar')
%% ��7.3�ȽϺ���wavefast��wacedec2��ִ��ʱ��
clc,clear,close all;
f=imread('Fig0704.tif');
[ratio,maxdifference]=fwtcompare(f,5,'db4')
%% ��7.4ʹ�ñ任�ֽ�����c��С�������亯��
clc,clear,close all;
f=magic(8);
[c1,s1]=wavedec2(f,3,'haar')
size(c1)
approx=appcoef2(c1,s1,'haar')
horizdet2=detcoef2('h',c1,s1,2)
newc1=wthcoef2('h',c1,s1,2);
newhorizdet2=detcoef2('h',newc1,s1,2)
%% ��7.5����wavecut��wavecopy����c
clc,clear,close all;
f=magic(8);
[c1,s1]=wavedec2(f,3,'haar');
approx=appcoef2(c1,s1,'haar')
horizdet2=detcoef2('h',c1,s1,2)
[newc1,horizdet2]=wavecut('h',c1,s1,2);
newhorizdet2=wavecopy('h',newc1,s1,2)
%% ��7.6��wavedispl������ʾ�任����
clc,clear,close all;
f=imread('Fig0704.tif');
[c,s]=wavefast(f,2,'db4');
figure
subplot(131),wave2gray(c,s);%�Զ��������Ŵ�
subplot(132),wave2gray(c,s,8);%8�������Ŵ�
subplot(133);wave2gray(c,s,-8);%������ֵ����8���Ŵ�
%% ��7.7��waveback��waverec2����ִ��ʱ��Ƚ�
clc,clear,close all;
f=imread('Fig0704.tif');
[ratio,maxdifference]=ifwtcompare(f,5,'db4')
%% ��7.8С���Ķ����Ժͱ�Ե���
clc,clear,close all;
f=imread('Fig0707(a).tif');
[c,s]=wavefast(f,1,'sym4');
[nc,y]=wavecut('a',c,s);
edges=abs(waveback(nc,s,'sym4'));
figure
subplot(221),imshow(f);
subplot(222),wavedisplay(c,s,6);%С���任
subplot(223),wavedisplay(nc,s,-6);%���н���ϵ������Ϊ0���޸ĺ�任
subplot(224),imshow(mat2gray(edges));%���㷴�任�ľ���ֵ�������õ��ı�Եͼ��
%% ��7.9����С��ͼ��ƽ����ģ��
clc,clear,close all;
f=imread('Fig0707(a).tif');
[c,s]=wavefast(f,4,'sym4');
figure
subplot(321),imshow(f);
subplot(322),wavedisplay(c,s,20);%С���任
subplot(323),[c,g8]=wavezero(c,s,1,'sym4');%����һ����ϸ��ϵ������Ϊ0�ķ��任
subplot(324),[c,g8]=wavezero(c,s,2,'sym4');
subplot(325),[c,g8]=wavezero(c,s,3,'sym4');
subplot(326),[c,g8]=wavezero(c,s,4,'sym4');
%% ��7.10�����ع�
clc,clear,close all;
f=imread('Fig0709(f).tif');
[c,s]=wavefast(f,4,'jpeg9.7');
wavedisplay(c,s,8);%4�߶�С���任

f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));%���Ͻǵĵ�4������ͼ��

[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));%�ϳɵ�4��ϸ�ڵľ�ȷ����
[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));
[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));
[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));



