%% 3.1已填充和未填充滤波的效果
clc,clear,close all;
f = imread('Fig0305(a).tif');
figure
subplot(131),imshow(f,[]);
[M, N] = size(f);
[f, revertclass] = tofloat(f);
F = fft2(f);
sig = 10;
H = lpfilter('gaussian', M, N, sig);
G = H.*F;
g = ifft2(G);
g = revertclass(g);
subplot(132),imshow(g)
PQ = paddedsize(size(f));
Fp = fft2(f, PQ(1), PQ(2));
Hp = lpfilter('gaussian', PQ(1), PQ(2), 2*sig);
Gp = Hp.*Fp;
gp = ifft2(Gp);
gpc = gp(1:size(f,1), 1:size(f,2));
gpc = revertclass(gpc);
subplot(133),imshow(gpc)
figure
imshow(gp)
h = fspecial('gaussian', 15, 7);
gs = imfilter(f, h);

%% 3.2空域和频域滤波器的比较
clc,clear,close all;
f = imread('Fig0309(a).tif');
figure
subplot(121),imshow(f,[])
f = tofloat(f);
F = fft2(f);
S = fftshift(log(1 + abs(F)));
subplot(122),imshow(S, [])
h = fspecial('sobel');
%freqz2(h)
PQ = paddedsize(size(f));
H = freqz2(h, PQ(1), PQ(2));
H1 = ifftshift(H);
figure
subplot(221),mesh(double(abs(H(1:10:1200,1:10:1200)))),axis tight,colormap([0 0 0]),axis off
subplot(222),mesh(double(abs(H1(1:10:1200,1:10:1200)))),axis tight,colormap([0 0 0]),axis off
subplot(223),imshow(abs(H), [])
subplot(224),imshow(abs(H1),[])
gs = imfilter(f, h);
gf = dftfilt(f, H1);
figure
subplot(331),imshow(gs, [])
subplot(322),imshow(gf,[])
subplot(323),imshow(abs(gs),[])
subplot(324),imshow(abs(gf),[])
subplot(325),imshow(abs(gs) > 0.2*abs(max(gs(:))))
subplot(326),imshow(abs(gf) > 0.2*abs(max(gf(:))))
d = abs(gs - gf);
max(d(:))
min(d(:))
%% 例3.3 dftuv的使用
clc,clear,close all;
[U, V] = dftuv(8, 5);
DSQ = U.^2 + V.^2
fftshift(DSQ)
fftshift(hypot(U,V)) %距离计算

%% 例3.4 低通滤波器
clc,clear,close all;
f = imread('Fig0313(a).tif');
[f, revertclass] = tofloat(f);
PQ = paddedsize(size(f));
[U, V] = dftuv(PQ(1), PQ(2));
D = hypot(U, V);
D0 = 0.05*PQ(2);
F = fft2(f,PQ(1),PQ(2));
H = exp(-(D.^2)/(2*(D0^2)));
g = dftfilt(f,H);
g = revertclass(g);
figure
subplot(221),imshow(f,[])
subplot(222),imshow(fftshift(H))
subplot(223),imshow(log(1 + abs(fftshift(F))),[])
subplot(224),imshow(g)

%% 例3.5 线框绘制
clc,clear,close all;
H = fftshift(lpfilter('gaussian', 500, 500, 50));
figure
subplot(221),mesh(double(H(1:10:500,1:10:500)))
axis tight
subplot(222),mesh(double(H(1:10:500,1:10:500)))
colormap([0 0 0])
axis off
subplot(223),mesh(double(H(1:10:500,1:10:500)))
colormap([0 0 0])
axis off
view(-25,30)
subplot(224),mesh(double(H(1:10:500,1:10:500)))
colormap([0 0 0])
axis off
view(-25,0)
figure
subplot(121),surf(double(H(1:10:500,1:10:500)))
axis tight
colormap(gray)
axis off
subplot(122),surf(double(H(1:10:500,1:10:500)))
axis tight
colormap(gray)
axis off
shading interp %平滑小面描影和消除栅栏网线
%函数图像绘制
[Y,X]=meshgrid(-2:0.1:2,-2:0.1:2);
Z=X.*exp(-X.^2-Y.^2);
figure,mesh(Z)
figure,surf(Z)
shading interp

%% 例3.6高通滤波
clc,clear,close all;
H1 = fftshift(hpfilter('ideal',500,500,50));
H2 = fftshift(hpfilter('btw',500,500,50));
H3 = fftshift(hpfilter('gaussian',500,500,50));
figure
subplot(231),mesh(double(H1(1:10:500,1:10:500)));
axis tight
colormap([0 0 0])
axis off
subplot(234),imshow(H1,[])
subplot(232),mesh(double(H2(1:10:500,1:10:500)));
axis tight
colormap([0 0 0])
axis off
subplot(235),imshow(H2,[])
subplot(233),mesh(double(H3(1:10:500,1:10:500)));
axis tight
colormap([0 0 0])
axis off
subplot(236),imshow(H3,[])
%% 例3.7 高通滤波的应用
clc,clear,close all;
f = imread('Fig0313(a).tif');
PQ= paddedsize(size(f));
D0 = 0.05*PQ(1);
H = hpfilter('gaussian',PQ(1),PQ(2),D0);
g = dftfilt(f,H);
figure
subplot(121),imshow(f)
subplot(122),imshow(g)
%% 例3.8 将高频强调滤波和直方图均衡结合起来
clc,clear,close all;
f=imread('Fig0303(a).tif');
figure,imshow(f)
F=fft2(f);
S=abs(F);
figure,imshow(S,[])
Fc=fftshift(F);
figure,imshow(Fc,[]);
figure,imshow(abs(Fc),[]);
S2=log(1+abs(Fc));
figure,imshow(S2,[]);
F=ifftshift(Fc);
phi=angle(F);
figure,imshow(phi,[]);

%% 例3.9 用陷波滤波器减少波纹模式
clc,clear,close all;
f=imread('Fig0321(a).tif');
figure,imshow(f)
[M N]=size(f);
[f,revertclass]=tofloat(f);
F=fft2(f);
S=gscale(log(1+abs(fftshift(F))));
figure,imshow(S)
C1=[99 154;128 163];
H1=cnotch('gaussian','reject',M,N,C1,5);
P1=gscale(fftshift(H1).*(tofloat(S)));
figure,imshow(P1)
g1=dftfilt(f,H1);
g1=revertclass(g1);
figure,imshow(g1)

C2=[99 154;128 163;49 160;133 133;55 132;108 255;112 74];
H2=cnotch('gaussian','reject',M,N,C2,5);
P2=gscale(fftshift(H2).*(tofloat(S)));
figure,imshow(P2)
g2=dftfilt(f,H2);
g2=revertclass(g2);
figure,imshow(g2)
%% 例3.10 陷波滤通器减少因有故障的成像设备而导致的周期干扰
clc,clear,close all;
f=imread('Fig0322(a).tif');
figure,imshow(f)
[M N]=size(f);
[f,revertclass]=tofloat(f);
F=fft2(f);
S=gscale(log(1+abs(fftshift(F))));
figure,imshow(S)
H=recnotch('reject','vertical',M,N,3,15,15);
figure,imshow(fftshift(H))
g=dftfilt(f,H);
g=revertclass(g);
figure,imshow(g)
%% 得到空间干扰模式自身
Hrecpass=recnotch('pass','vertical',M,N,3,15,15);
interference=dftfilt(f,Hrecpass);
figure,imshow(fftshift(Hrecpass))
interference=gscale(interference);
figure,imshow(interference)
