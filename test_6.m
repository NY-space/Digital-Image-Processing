%% ��6.1�Ը�������ĳЩ�������÷���ʾ
clc,clear,close all;
f = imread('Fig0604(a).tif');
[X1, map1] = rgb2ind(f, 8, 'nodither');%������ɫ����������
figure
subplot(231),imshow(f)
subplot(232),imshow(X1, map1)
[X2, map2] = rgb2ind(f, 8, 'dither');
subplot(233),imshow(X2, map2)
g = rgb2gray(f);%�õ��Ҷ�ͼ��
g1 = dither(g);
subplot(234),imshow(g)
subplot(235),imshow(g1)
%% ��6.2��RGBת��ΪHSI
clc,clear,close all;
f = imread('Fig0602(b).tif');
hsi = rgb2hsi(f);
figure
subplot(221),imshow(f(:,:,1:3))
subplot(222),imshow(hsi(:,:,1))%ɫ��ͼ��
subplot(223),imshow(hsi(:,:,2))%���Ͷ�ͼ��
subplot(224),imshow(hsi(:,:,3))%����ͼ��
%% ��6.3��L*a*b*��ɫ�ռ�Ϊ���������ڸо���һ�µĲ�ɫ�ռ�
clc,clear,close all;
L = linspace(40, 80, 1024);%40��80��ȼ������1024��
radius = 70;
theta = linspace(0, pi, 1024);
a = radius * cos(theta);
b = radius * sin(theta);
L = repmat(L, 100, 1);
a = repmat(a, 100, 1);
b = repmat(b, 100, 1);
lab_scale = cat(3, L, a, b);%��ά��ʽ�ѵ�������ɫͼ��
cform = makecform('lab2srgb');
rgb_scale = applycform(lab_scale, cform);%ת��ΪRGB
imshow(rgb_scale)
%% ��6.4ICC��ɫ�������ʵ��
clc,clear,close all;
f = imread('Fig0604(a).tif');
fp = padarray(f, [40 40], 255, 'both');%��Ӱױ�
fp = padarray(fp, [4 4], 230, 'both');%���ϸ�ұ�
figure
subplot(121),imshow(fp)
p_srgb = iccread('sRGB.icm');
p_snap = iccread('SNAP2007.icc');
cform1 = makecform('icc', p_srgb, p_snap);
fp_newsprint = applycform(fp, cform1);
%����ɫ����Ⱦ
cform2 = makecform('icc', p_snap, p_srgb,'SourceRenderingIntent', 'AbsoluteColorimetric','DestRenderingIntent', 'AbsoluteColorimetric');
fp_proof = applycform(fp_newsprint, cform2);
subplot(122),imshow(fp_proof)
%% ��6.5��ɫ��Ƭ�Ͳ�ɫ�����ķ�ӳ��
clc,clear,close all;
f = imread('Fig0615[original].tif');
ice('image', f);
f2 = imread('Fig0617(a).tif');
ice('image', f2);
%% ��6.6��ɫ�Ͳ�ɫ�Ա���ǿ
clc,clear,close all;
f = imread('Fig0618(a).tif');
ice('image', f);
f2 = imread('Fig0618(d).tif');
ice('image', f2);
%% ��6.7α��ɫӳ��
clc,clear,close all;
f = imread('Fig0619(a).tif');
ice('image', f);
%% ��6.8ɫ��ƽ��
clc,clear,close all;
f = imread('Fig0620(a).tif');
ice('image', f, 'space', 'CMY');
%% ��6.9����ֱ��ͼ��ӳ��
clc,clear,close all;
f = imread('Fig0621(a).tif');
ice('image', f);
%% ��6.10��ɫͼ���ƽ������
clc,clear,close all;
f = imread('Fig0622(a).tif');
fR = f(:,:,1);%��ȡ����ͼ��
fG = f(:,:,2);
fB = f(:,:,3);
figure
subplot(141),imshow(f)
subplot(142),imshow(fR)
subplot(143),imshow(fG)
subplot(144),imshow(fB)
w = fspecial('average', 25);
fR_filtered = imfilter(fR, w, 'replicate');%ƽ���˲����ֱ����
fG_filtered = imfilter(fG, w, 'replicate');
fB_filtered = imfilter(fB, w, 'replicate');
f_filtered = cat(3,fR_filtered,fG_filtered,fB_filtered);%�ؽ�RGB
h = rgb2hsi(f);%�õ�HSI����ͼ��
H = h(:,:,1);
S = h(:,:,2);
I = h(:,:,3);
figure
subplot(131),imshow(H)
subplot(132),imshow(S)
subplot(133),imshow(I)
I_filtered = imfilter(I,w,'replicate')%�˲����ȷ���;
h = cat(3,H,S,I_filtered);
f = hsi2rgb(h);%�õ�RGBͼ��
figure
subplot(131),imshow(f_filtered)
subplot(132),imshow(f)
H_filtered = imfilter(H,w,'replicate');
S_filtered = imfilter(S,w,'replicate');
h = cat(3,H_filtered,S_filtered,I_filtered);
f = hsi2rgb(h);
subplot(133),imshow(f)
%% ��6.11ɫ��ͼ����񻯴���
clc,clear,close all;
fb = imread('Fig0625(a).tif');
lapmask = [1 1 1;1 -8 1;1 1 1];%������˹�˲�ģ��
fb = tofloat(fb);
fen = fb - imfilter(fb, lapmask, 'replicate');
figure
subplot(121),imshow(fb)
subplot(122),imshow(fen)
%% ��6.12ʹ�ú���colorgrad���RGB��Ե
clc,clear,close all;
a=imread('Fig0627(a).tif');
b=imread('Fig0627(b).tif');
c=imread('Fig0627(c).tif');
f=cat(3,a,b,c);
[VG,A,PPG]=colorgrad(f);
figure
subplot(231),imshow(a)
subplot(232),imshow(b)
subplot(233),imshow(c)
subplot(234),imshow(f)
subplot(235),imshow(VG)%�����ռ���ֱ�Ӽ����ݶ�
subplot(236),imshow(PPG)%�ֱ����RGB����ͼ���2D�ݶȲ����
g=imread('Fig0628(a).tif');
[VG,A,PPG]=colorgrad(g);
g1=mat2gray(abs(VG-PPG));
figure
subplot(221),imshow(g)
subplot(222),imshow(VG)
subplot(223),imshow(PPG)
subplot(224),imshow(g1)
%% ��6.13RGB��ɫͼ��ָ�
clc,clear,close all;
f=imread('Fig0630(a).tif');
mask=roipoly(f);%�����ȡ
red=immultiply(mask,f(:,:,1));
green=immultiply(mask,f(:,:,2));
blue=immultiply(mask,f(:,:,3));
g=cat(3,red,green,blue);
figure,imshow(g)
[M,N,K]=size(g);
I=reshape(g,M*N,3);
idx=find(mask);
I=double(I(idx,1:3));
[C,m]=covmatrix(I);
d=diag(C);
sd=sqrt(d);
E25=colorseg('euclidean',f,25,m);
E50=colorseg('euclidean',f,50,m);
E75=colorseg('euclidean',f,75,m);
E100=colorseg('euclidean',f,100,m);
figure
subplot(221),imshow(E25)
subplot(222),imshow(E50)
subplot(223),imshow(E75)
subplot(224),imshow(E100)

E25_=colorseg('mahalanobis',f,25,m,C);
E50_=colorseg('mahalanobis',f,50,m,C);
E75_=colorseg('mahalanobis',f,75,m,C);
E100_=colorseg('mahalanobis',f,100,m,C);
figure
subplot(221),imshow(E25_)
subplot(222),imshow(E50_)
subplot(223),imshow(E75_)
subplot(224),imshow(E100_)