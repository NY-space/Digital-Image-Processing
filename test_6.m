%% 例6.1对给定表中某些函数的用法表示
clc,clear,close all;
f = imread('Fig0604(a).tif');
[X1, map1] = rgb2ind(f, 8, 'nodither');%八种颜色，抖动处理
figure
subplot(231),imshow(f)
subplot(232),imshow(X1, map1)
[X2, map2] = rgb2ind(f, 8, 'dither');
subplot(233),imshow(X2, map2)
g = rgb2gray(f);%得到灰度图像
g1 = dither(g);
subplot(234),imshow(g)
subplot(235),imshow(g1)
%% 例6.2从RGB转换为HSI
clc,clear,close all;
f = imread('Fig0602(b).tif');
hsi = rgb2hsi(f);
figure
subplot(221),imshow(f(:,:,1:3))
subplot(222),imshow(hsi(:,:,1))%色调图像
subplot(223),imshow(hsi(:,:,2))%饱和度图像
subplot(224),imshow(hsi(:,:,3))%亮度图像
%% 例6.3以L*a*b*彩色空间为基础创建在感觉上一致的彩色空间
clc,clear,close all;
L = linspace(40, 80, 1024);%40到80间等间隔插入1024点
radius = 70;
theta = linspace(0, pi, 1024);
a = radius * cos(theta);
b = radius * sin(theta);
L = repmat(L, 100, 1);
a = repmat(a, 100, 1);
b = repmat(b, 100, 1);
lab_scale = cat(3, L, a, b);%三维方式堆叠产生彩色图像
cform = makecform('lab2srgb');
rgb_scale = applycform(lab_scale, cform);%转换为RGB
imshow(rgb_scale)
%% 例6.4ICC彩色剖面的软实验
clc,clear,close all;
f = imread('Fig0604(a).tif');
fp = padarray(f, [40 40], 255, 'both');%添加白边
fp = padarray(fp, [4 4], 230, 'both');%添加细灰边
figure
subplot(121),imshow(fp)
p_srgb = iccread('sRGB.icm');
p_snap = iccread('SNAP2007.icc');
cform1 = makecform('icc', p_srgb, p_snap);
fp_newsprint = applycform(fp, cform1);
%绝度色度渲染
cform2 = makecform('icc', p_snap, p_srgb,'SourceRenderingIntent', 'AbsoluteColorimetric','DestRenderingIntent', 'AbsoluteColorimetric');
fp_proof = applycform(fp_newsprint, cform2);
subplot(122),imshow(fp_proof)
%% 例6.5单色负片和彩色分量的反映射
clc,clear,close all;
f = imread('Fig0615[original].tif');
ice('image', f);
f2 = imread('Fig0617(a).tif');
ice('image', f2);
%% 例6.6单色和彩色对比增强
clc,clear,close all;
f = imread('Fig0618(a).tif');
ice('image', f);
f2 = imread('Fig0618(d).tif');
ice('image', f2);
%% 例6.7伪彩色映射
clc,clear,close all;
f = imread('Fig0619(a).tif');
ice('image', f);
%% 例6.8色彩平衡
clc,clear,close all;
f = imread('Fig0620(a).tif');
ice('image', f, 'space', 'CMY');
%% 例6.9基于直方图的映射
clc,clear,close all;
f = imread('Fig0621(a).tif');
ice('image', f);
%% 例6.10彩色图像的平滑处理
clc,clear,close all;
f = imread('Fig0622(a).tif');
fR = f(:,:,1);%抽取分量图像
fG = f(:,:,2);
fB = f(:,:,3);
figure
subplot(141),imshow(f)
subplot(142),imshow(fR)
subplot(143),imshow(fG)
subplot(144),imshow(fB)
w = fspecial('average', 25);
fR_filtered = imfilter(fR, w, 'replicate');%平滑滤波器分别过滤
fG_filtered = imfilter(fG, w, 'replicate');
fB_filtered = imfilter(fB, w, 'replicate');
f_filtered = cat(3,fR_filtered,fG_filtered,fB_filtered);%重建RGB
h = rgb2hsi(f);%得到HSI分量图像
H = h(:,:,1);
S = h(:,:,2);
I = h(:,:,3);
figure
subplot(131),imshow(H)
subplot(132),imshow(S)
subplot(133),imshow(I)
I_filtered = imfilter(I,w,'replicate')%滤波亮度分量;
h = cat(3,H,S,I_filtered);
f = hsi2rgb(h);%得到RGB图像
figure
subplot(131),imshow(f_filtered)
subplot(132),imshow(f)
H_filtered = imfilter(H,w,'replicate');
S_filtered = imfilter(S,w,'replicate');
h = cat(3,H_filtered,S_filtered,I_filtered);
f = hsi2rgb(h);
subplot(133),imshow(f)
%% 例6.11色彩图像的锐化处理
clc,clear,close all;
fb = imread('Fig0625(a).tif');
lapmask = [1 1 1;1 -8 1;1 1 1];%拉普拉斯滤波模板
fb = tofloat(fb);
fen = fb - imfilter(fb, lapmask, 'replicate');
figure
subplot(121),imshow(fb)
subplot(122),imshow(fen)
%% 例6.12使用函数colorgrad检测RGB边缘
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
subplot(235),imshow(VG)%向量空间中直接计算梯度
subplot(236),imshow(PPG)%分别计算RGB分量图像的2D梯度并相加
g=imread('Fig0628(a).tif');
[VG,A,PPG]=colorgrad(g);
g1=mat2gray(abs(VG-PPG));
figure
subplot(221),imshow(g)
subplot(222),imshow(VG)
subplot(223),imshow(PPG)
subplot(224),imshow(g1)
%% 例6.13RGB彩色图像分割
clc,clear,close all;
f=imread('Fig0630(a).tif');
mask=roipoly(f);%区域获取
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