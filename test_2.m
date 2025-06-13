%% 例2.1 使用imadjust函数
clc,clear,close all;
f = imread('Fig0203(a).tif');
g1 = imadjust(f, [0 1], [1 0]);
figure
subplot(231)
imshow(f,[])
subplot(232)
imshow(g1,[])
g2 = imadjust(f, [0.5 0.75], [0 1]);
g3 = imadjust(f, [], [], 2);
subplot(233)
imshow(g2,[])
subplot(234)
imshow(g3,[])
g4 = imadjust(f, stretchlim(f),[]);
g5 = imadjust(f, stretchlim(f),[1 0]);
subplot(235)
imshow(g4,[]);
subplot(236)
imshow(g5,[])
%% 例2.2利用对数变换减少动态范围
clc,clear,close all;
f = imread('Fig0205(a).tif');
g = im2uint8(mat2gray(log(1 + double(f))));
figure
subplot(121)
imshow(f,[])
subplot(122)
imshow(g,[])
figure('Name','傅里叶频谱-对数变换','NumberTitle','off'),imshow(g)
%% 例2.3针对intrans函数的说明
clc,clear,close all;
f = imread('Fig0206(a).tif');
g = intrans(f, 'stretch', mean2(tofloat(f)),0.9);%对比度拉伸
figure
subplot(121),imshow(f,[])
subplot(122),imshow(g,[])
%% 例2.4计算并绘制图像直方图
clc,clear,close all;
%plot函数经常被用作显示变换函数
f = imread('Fig0203(a).tif');
figure
subplot(221),imhist(f)
h = imhist(f,25);
horz = linspace(0,255,25);
subplot(222),bar(horz,h)
axis([0 255 0 60000])
set(gca, 'xtick', 0:50:255)
set(gca, 'ytick', 0:20000:60000)
subplot(223),stem(horz,h,'fill')
axis([0 255 0 60000])
set(gca, 'xtick', 0:50:255)
set(gca, 'ytick', 0:20000:60000)
hc = imhist(f);
subplot(224),plot(hc)
axis([0 255 0 15000])
set(gca, 'xtick', 0:50:255)
set(gca, 'ytick', 0:5000:15000)
%处理函数句柄，常用函数fplot绘图
fhandle=@tanh;
figure,fplot(fhandle,[-2 2],'g-P' )
%% 例2.5直方图均衡化
clc,clear,close all;
f = imread('Fig0208(a).tif');
figure
subplot(221),imshow(f,[]);
g = histeq(f,256);
subplot(222),imhist(f),ylim('auto');
subplot(223),imshow(g,[]);
subplot(224),imhist(g),ylim('auto');
hnorm = imhist(f)./numel(f);
cdf = cumsum(hnorm);%归一化直方图的累加求和的变换函数
figure
x = linspace(0,1,256);
plot(x,cdf)
axis([0 1 0 1]);
set(gca,'xtick',0:.2:1)
set(gca,'ytick',0:.2:1)
xlabel('Input intensity values','fontsize',9)
ylabel('Output intensity values','fontsize',9)
%% 例2.6直方图匹配
clc,clear,close all;
f = imread('Fig0210(a).tif');
f1 = histeq(f,256);
figure
subplot(221),imshow(f,[]);
subplot(222),imhist(f),ylim('auto');
subplot(223),imshow(f1,[]);
subplot(224),imhist(f1),ylim('auto');
p = twomodegauss(0.15,0.05,0.75,0.05,1,0.07,0.002);%规定的直方图
figure
subplot(221),plot(p),xlim([0 255]);
g = histeq(f, p);
subplot(222),imshow(g);
subplot(223),imhist(g),ylim('auto');
%% 例2.7adapthisteq函数的使用（对比度受限的自适应直方图均衡，小区域处理再双线性内插组合）
clc,clear,close all;
f = imread('Fig0210(a).tif');
g1 = adapthisteq(f);
g2 = adapthisteq(f,'NumTiles',[25 25]);
g3 = adapthisteq(f, 'NumTiles',[25 25],'ClipLimit',0.05);
figure
subplot(141),imshow(f,[])
subplot(142),imshow(g1,[])
subplot(143),imshow(g2,[])
subplot(144),imshow(g3,[])
%% 2.8函数imfilter的应用(默认相关，卷积需将滤波器w旋转180度，用rot90(w,2)实现)
clc,clear,close all;
w = ones(31);%对称
f = imread('Fig0216(a).tif');
figure
subplot(231),imshow(f,[])
gd = imfilter(f,w);%默认填充0
subplot(232),imshow(gd,[]);
gr = imfilter(f,w,'replicate');%填充外边界
subplot(233),imshow(gr,[])
gs = imfilter(f,w,'symmetric');%填充镜像
subplot(234),imshow(gs,[]);
gc = imfilter(f,w,'circular');%填充周期
subplot(235),imshow(gc,[]);
f8 = im2uint8(f);
g8r = imfilter(f8,w,'replicate');
subplot(236),imshow(g8r,[])
%% 例2.9使用函数colfilt实现非线性空间滤波
clc,clear,close all;
f = [1 2;3 4];
gmean = @(A) prod(A, 1).^(1./ size(A,1)); %prod计算乘积，矩阵计算每列乘积
f1 = padarray(f , [2 2], 'replicate');
g =colfilt(f1,[2 2],'sliding',gmean);
[M N]=size(f);
g=g((1:M)+2,(1:N)+2)
%% 例2.10 使用函数imfilter实现拉普拉斯滤波器（图像锐化，叠加恢复灰度色级）
clc,clear,close all;
f = imread('Fig0217(a).tif');
w = fspecial('laplacian',0)
g1 = imfilter(f,w,'replicate');
figure
subplot(221),imshow(f,[])
subplot(222),imshow(g1,[])%负值被截掉
f2 = tofloat(f);
g2=imfilter(f2,w,'replicate');
subplot(223),imshow(g2,[])
g = f2-g2;%w中心值为负用减
subplot(224),imshow(g);
%% 例2.11 人工规定的滤波器与增强技术
clc,clear,close all;
f = imread('Fig0217(a).tif');
w4 = fspecial('laplacian',0);
w8 = [1 1 1;1 -8 1;1 1 1];
f = tofloat(f);
g4 = f - imfilter(f,w4,'replicate');
g8 = f - imfilter(f,w8,'replicate');
figure
subplot(131),imshow(f)
subplot(132),imshow(g4)
subplot(133),imshow(g8)
%% 例2.12利用函数medfilt2的中值滤波（减少椒盐噪声）
clc,close,close all;
f = imread('Fig0219(a).tif');
fn = imnoise(f,'salt & pepper',0.2);
gm = medfilt2(fn);
gms = medfilt2(fn,'symmetric');%，在边界处对称延伸图像，消除黑色边缘效应
figure
subplot(221),imshow(f)
subplot(222),imshow(fn)
subplot(223),imshow(gm)
subplot(224),imshow(gms)
%% 例2.13对模糊集合定义的说明
%% 例2.14使用模糊函数
clc,close,close all;
ulow=@(z) 1-sigmamf(z,0.27,0.47);
umid=@(z) triangmf(z,0.24,0.50,0.74);
uhigh=@(z) sigmamf(z,0.53,0.73);
fplot(ulow,[0 1],20);
hold on
fplot(umid,[0 1],'-.',20);
fplot(uhigh,[0 1],'--',20);
hold off
title('Input membership funcftions ,Example 2.14')
unorm=@(z) 1-sigmamf(z,0.18,0.33);
umarg=@(z) trapezmf(z,0.23,0.35,0.53,0.69);
ufail=@(z) sigmamf(z,0.59,0.78);
rules={ulow;umid;uhigh};
L=lambdafcns(rules);
z=0.7;
outputmfs={unorm,umarg,ufail};
Q=implfcns(L,outputmfs,z);
Qa=aggfcn(Q);%聚合
final_result=defuzzify(Qa,[0 1])%去模糊
F=fuzzysysfcn(rules,outputmfs,[0 1]);
F(0.7)
%使用approxfcn函数
G=approxfcn(F,[0 1]);
G(0.7)
fplot(F,[0 1],'k',20)
hold on
fplot(G,[0 1],'k:o',20)
hold off
f=@() F(0.7);
g=@() G(0.7);
t1=timeit(f);
t2=timeit(g);
t=t1/t2 %近似模糊系统函数运行更快
%% 例2.15用模糊函数实现模糊对比度增强
clc,close,close all;
f=imread('Fig0228(a).tif');
udark=@(z) 1-sigmamf(z,0.35,0.5);
ugray=@(z) triangmf(z,0.35,0.5,0.65);
ubright=@(z) sigmamf(z,0.5,0.65);
figure
fplot(udark,[0 1],20)
hold on
fplot(ugray,[0 1],20)
fplot(ubright,[0 1],20)
udarker=@(z) bellmf(z,0.0,0.1);
umidgray=@(z) bellmf(z,0.4,0.5);
ubrighter=@(z) bellmf(z,0.8,0.9);
rules={udark;ugray;ubright};
outmf={udarker,umidgray,ubrighter};
F=fuzzysysfcn(rules,outmf,[0 1]);
z=linspace(0,1,256);
T=F(z);
g1=intrans(f,'specified',T);
figure
subplot(231),imshow(f,[]);
subplot(234),imhist(f);
g = histeq(f,256);
subplot(232),imshow(g,[]);
subplot(235),imhist(g);
subplot(233),imshow(g1,[]);
subplot(236),imhist(g1);
%% 例2.16使用模糊的基本规则的空间滤波检测边界
clc,close,close all;
f=imread('Fig0233(a).tif');
figure
subplot(131),imshow(f)
f1=fuzzyfilt(f);
subplot(132),imshow(f1)
f2=mat2gray(f1);
subplot(133),imshow(f2)





