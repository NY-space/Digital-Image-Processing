%% 例7.1Harr滤波器、尺度和小波函数
clc,clear,close all;
[Lo_D,Hi_d,Lo_R,hi_R]=wfilters('haar')
waveinfo('haar');
[phi,psi,xval]=wavefun('haar',10);
xaxis = zeros(size(xval));
subplot(121);plot(xval,phi,'k',xval,xaxis,'--k');
axis([0 1 -1.5 1.5]);axis square;
title('Haar尺度函数');
subplot(122);plot(xval,psi,'k',xval,xaxis,'--k');
axis([0 1 -1.5 1.5]);axis square;
title('Harr小波函数');
%% 例7.2使用Haar滤波器的简单快速小波变换（FWT）
clc,clear,close all;
f=magic(4)%魔方矩阵，行列对角线元素之和相等
[c1,s1]=wavedec2(f,1,'haar')
[c2,s2]=wavedec2(f,2,'haar')
%% 例7.3比较函数wavefast和wacedec2的执行时间
clc,clear,close all;
f=imread('Fig0704.tif');
[ratio,maxdifference]=fwtcompare(f,5,'db4')
%% 例7.4使用变换分解向量c的小波工具箱函数
clc,clear,close all;
f=magic(8);
[c1,s1]=wavedec2(f,3,'haar')
size(c1)
approx=appcoef2(c1,s1,'haar')
horizdet2=detcoef2('h',c1,s1,2)
newc1=wthcoef2('h',c1,s1,2);
newhorizdet2=detcoef2('h',newc1,s1,2)
%% 例7.5运用wavecut和wavecopy处理c
clc,clear,close all;
f=magic(8);
[c1,s1]=wavedec2(f,3,'haar');
approx=appcoef2(c1,s1,'haar')
horizdet2=detcoef2('h',c1,s1,2)
[newc1,horizdet2]=wavecut('h',c1,s1,2);
newhorizdet2=wavecopy('h',newc1,s1,2)
%% 例7.6用wavedispl函数显示变换函数
clc,clear,close all;
f=imread('Fig0704.tif');
[c,s]=wavefast(f,2,'db4');
figure
subplot(131),wave2gray(c,s);%自动按比例放大
subplot(132),wave2gray(c,s,8);%8倍比例放大
subplot(133);wave2gray(c,s,-8);%按绝对值进行8倍放大
%% 例7.7对waveback和waverec2函数执行时间比较
clc,clear,close all;
f=imread('Fig0704.tif');
[ratio,maxdifference]=ifwtcompare(f,5,'db4')
%% 例7.8小波的定向性和边缘检测
clc,clear,close all;
f=imread('Fig0707(a).tif');
[c,s]=wavefast(f,1,'sym4');
[nc,y]=wavecut('a',c,s);
edges=abs(waveback(nc,s,'sym4'));
figure
subplot(221),imshow(f);
subplot(222),wavedisplay(c,s,6);%小波变换
subplot(223),wavedisplay(nc,s,-6);%所有近似系数设置为0的修改后变换
subplot(224),imshow(mat2gray(edges));%计算反变换的绝对值，进而得到的边缘图像
%% 例7.9基于小波图像平滑及模糊
clc,clear,close all;
f=imread('Fig0707(a).tif');
[c,s]=wavefast(f,4,'sym4');
figure
subplot(321),imshow(f);
subplot(322),wavedisplay(c,s,20);%小波变换
subplot(323),[c,g8]=wavezero(c,s,1,'sym4');%将第一级的细节系数设置为0的反变换
subplot(324),[c,g8]=wavezero(c,s,2,'sym4');
subplot(325),[c,g8]=wavezero(c,s,3,'sym4');
subplot(326),[c,g8]=wavezero(c,s,4,'sym4');
%% 例7.10渐进重构
clc,clear,close all;
f=imread('Fig0709(f).tif');
[c,s]=wavefast(f,4,'jpeg9.7');
wavedisplay(c,s,8);%4尺度小波变换

f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));%左上角的第4级近似图像

[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));%合成第4级细节的精确近似
[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));
[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));
[c,s]=waveback(c,s,'jpeg9.7',1);
f=wavecopy('a',c,s);
figure;imshow(mat2gray(f));



