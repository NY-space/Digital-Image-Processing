%% 例10.1点检测
clc,clear,close all;
f = imread('Fig1002(a).tif');
w = [-1 -1 -1; -1 8 -1; -1 -1 -1];
g = abs(imfilter(tofloat(f),w));
T = max(g(:));
g = g >= T;
figure
subplot(121),imshow(f)
subplot(122),imshow(g)
%% 例10.2线检测
clc,clear,close all;
f = imread('Fig1004(a).tif');
w = [2 -1 -1; -1 2 -1; -1 -1 2];
g = imfilter(tofloat(f),w);
figure
subplot(321),imshow(f)
subplot(322),imshow(g,[])
gtop = g(1:120,1:120);
gtop = pixeldup(gtop,4);
subplot(323),imshow(gtop,[])
gbot = g(end - 119:end, end - 119:end);
gbot = pixeldup(gbot, 4);
subplot(324),imshow(gbot, [])
g = abs(g);
subplot(325),imshow(g, [])
T = max(g(:));
g = g >= T;
subplot(326),imshow(g)
%% 例10.3Sobel边缘检测算子的使用
clc,clear,close all;
f = imread('Fig1006(a).tif');
[gv, t] = edge(f, 'sobel', 'vertical');%垂直边缘检测
figure
subplot(321),imshow(f)
subplot(322),imshow(gv)
t
gv =edge(f, 'sobel', 0.15, 'vertical');%设定阈值的垂直边缘检测
subplot(323),imshow(gv)
gboth = edge(f, 'sobel', 0.15);%设定阈值的垂直和水平边缘检测
subplot(324),imshow(gboth)
wneg45 = [-2 -1 0; -1 0 1; 0 1 2]%-45°边缘检测
gneg45 = imfilter(tofloat(f), wneg45, 'replicate');
T = 0.3*max(abs(gneg45(:)));
gneg45 = gneg45 >= T;
subplot(325),imshow(gneg45)
wpos45 = [0 1 2; -1 0 1; -2 -1 0]%+45°边缘检测
gpos45 = imfilter(tofloat(f), wpos45, 'replicate');
T = 0.3*max(abs(gpos45(:)));
gpos45 = gpos45 >= T;
subplot(326),imshow(gpos45)
%% 例10.4Sobel,LoG和Canny边缘检测算子的比较
clc,clear,close all;
f = imread('Fig1006(a).tif');
f = tofloat(f);
[gSobel_default, ts] = edge(f, 'sobel');
[gLoG_default, ts] = edge(f, 'log');
[gCanny_default, ts] = edge(f, 'canny');
gSobel_best = edge(f, 'sobel', 0.05);
gLoG_best = edge(f, 'log', 0.003, 2.25);
gCanny_best = edge(f, 'canny', [0.04 0.10], 1.5);
figure
subplot(321),imshow(gSobel_default)%左列默认值，右列设置参数
subplot(322),imshow(gSobel_best)
subplot(323),imshow(gLoG_default)
subplot(324),imshow(gLoG_best)
subplot(325),imshow(gCanny_default)
subplot(326),imshow(gCanny_best)
%% 例10.5霍夫变换的说明
clc,clear,close all;
f = zeros(101, 101);
f(1,1) = 1;f(101,1) = 1; f(1,101) = 1;
f(101, 101) = 1; f(51,51) = 1;
H = hough(f);
subplot(221),imshow(f,[])
subplot(222),imshow(H, [])
[H, theta, rho] = hough(f);
subplot(223),imshow(H,[],'XData',theta,'YData',rho,'InitialMagnification','fit')
axis on,axis normal
xlabel('\theta'),ylabel('\rho')
%% 例10.6用霍夫变换检测和连接线
clc,clear,close all;
f = imread('Fig1006(a).tif');
f = tofloat(f);
f = edge(f, 'canny', [0.04 0.10], 1.5);
[H, theta, rho] = hough(f, 'ThetaResolution', 0.2);
figure
subplot(121),imshow(H, [], 'XData',theta,'YData',rho,'InitialMagnification','fit')
axis on,axis normal
xlabel('\theta'),ylabel('\rho')
peaks = houghpeaks(H, 5);%霍夫变换的峰值
hold on
plot(theta(peaks(:,2)),rho(peaks(:,1)),...
    'linestyle', 'none', 'marker', 's', 'color', 'w')
lines = houghlines(f, theta, rho, peaks);
subplot(122),imshow(f),hold on
for k = 1:length(lines)
    xy = [lines(k).point1 ; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color',[.8 .8 .8]);
end
%% 例10.7计算全局阈值
clc,clear,close all;
f = imread('Fig1013(a).tif');
count = 0;
T = mean2(f);
done = false;
while ~done
    count = count +1;
    g = f > T;
    Tnext = 0.5*(mean(f(g)) + mean(f(~g)));
    done = abs(T - Tnext) < 0.5;
    T = Tnext;
end
count
T
g = im2bw(f, T/255);
subplot(131),imshow(f)
subplot(132),imhist(f)
subplot(133),imshow(g)
%% 例10.8对使用Otsu's方法和10.3.2节的基本全局阈值处理方法分割图像的比较
clc,clear,close all;
f = imread('Fig1014(a).tif');
figure
subplot(221),imshow(f)
subplot(222),imhist(f)
count = 0;
T = mean2(f);
done = false;
while ~done
    count = count +1;
    g = f > T;
    Tnext = 0.5*(mean(f(g)) + mean(f(~g)));
    done = abs(T - Tnext) < 0.5;
    T = Tnext;
end
f2 = im2bw(f, T/255);
subplot(223),imshow(f2)
[T, SM] = graythresh(f);
g = im2bw(f,T);
subplot(224),imshow(g)
%% 在阈值处理之前先对图像进行平滑
clc,clear,close all;
f = imread('Fig1015(a)[noiseless].tif');
fn = imnoise(f, 'gaussian', 0, 0.038);
figure
subplot(231),imshow(fn)
subplot(232),imhist(fn)
Tn = graythresh(fn);
gn = im2bw(fn, Tn);
subplot(233),imshow(gn)
w = fspecial('average', 5);
fa = imfilter(fn, w, 'replicate');
subplot(234),imshow(fa)
subplot(235),imhist(fa)
Ta = graythresh(fa);
ga = im2bw(fa, Ta);
subplot(236),imshow(ga)
%% 例10.9使用基于梯度的边缘信息改进全局阈值处理
clc,clear,close all;
f = tofloat(imread('Fig1016(a).tif'));
figure
subplot(231),imshow(f)
subplot(232),imhist(f)
sx = fspecial('sobel');
sy = sx';
gx = imfilter(f, sx, 'replicate');
gy = imfilter(f, sy, 'replicate');
grad = sqrt(gx.*gx + gy.*gy);
grad = grad/max(grad(:));
h = imhist(grad);
Q = percentile2i(h, 0.999);
markerImage = grad > Q;
subplot(233),imshow(markerImage)
fp = f.*markerImage;
subplot(234),imshow(fp)
hp = imhist(fp);
hp(1) = 0;
subplot(235),bar(hp)
T = otsuthresh(hp);
g = im2bw(f, T);
subplot(236),imshow(g)
%% 例10.10用拉普拉斯边缘信息改进全局阈值处理
clc,clear,close all;
f = tofloat(imread('Fig1017(a).tif'));
figure
subplot(231),imshow(f)
subplot(232),imhist(f)
hf = imhist(f);
[Tf SMF] = graythresh(f);
gf = im2bw(f, Tf);
subplot(233),imshow(gf)
w = [-1 -1 -1; -1 8 -1; -1 -1 -1];
lap = abs(imfilter(f,w,'replicate'));
lap = lap/max(lap(:));
h = imhist(lap);
Q = percentile2i(h, 0.995);%输出灰度级
markerImage = lap > Q;
fp = f.*markerImage;
subplot(234),imshow(fp)
hp = imhist(fp);
hp(1) = 0;
subplot(235),bar(hp)
T = otsuthresh(hp);
g = im2bw(f,T);
subplot(236),imshow(g)
%% 例10.11对全局和局部阈值处理的比较
clc,clear,close all;
f = tofloat(imread('Fig1017(a).tif'));
figure
subplot(221),imshow(f)
[TGlobal] = graythresh(f);
gGlobal = im2bw(f, TGlobal);
subplot(222),imshow(gGlobal)
g = localthresh(f, ones(3), 30, 1.5, 'global');
SIG = stdfilt(f, ones(3));
subplot(223),imshow(SIG, [])
subplot(224),imshow(g)
%% 例10.12利用移动平均的图像阈值处理
clc,clear,close all;
f = imread('Fig1019(a).tif');
T = graythresh(f);
g1 = im2bw(f, T);
g2 = movingthresh(f, 20, 0.5);
figure
subplot(231),imshow(f)
subplot(232),imshow(g1)
subplot(233),imshow(g2)
f = imread('Fig1019(d).tif');
T = graythresh(f);
g1 = im2bw(f, T);
g2 = movingthresh(f, 20, 0.5);
subplot(234),imshow(f)
subplot(235),imshow(g1)
subplot(236),imshow(g2)
%% 例10.13使用区域生长检测焊接空隙
clc,clear,close all;
f = imread('Fig1020(a).tif');
[g, NR, SI, TI] = regiongrow(f, 1, 0.26);
figure
subplot(221),imshow(f)
subplot(222),imshow(SI)%种子点
subplot(223),imshow(TI)%显示所以通过阈值测试的点
subplot(224),imshow(g)
figure
imhist(f)
ylim('auto')
%% 例10.14使用了区域分离和合并的图像分割
clc,clear,close all;
f = imread('Fig1023(a).tif');
g1 = splitmerge(f,64,@predicate);
g2 = splitmerge(f,32,@predicate);
g3 = splitmerge(f,16,@predicate);
g4 = splitmerge(f,8,@predicate);
g5 = splitmerge(f,4,@predicate);
figure
subplot(231),imshow(f)
subplot(232),imshow(g1)
subplot(233),imshow(g2)
subplot(234),imshow(g3)
subplot(235),imshow(g4)
subplot(236),imshow(g5)
%% 例10.15使用距离变换和分水岭变换分割二值图像
clc,clear,close all;
f= tofloat(~imread('Fig1026(a).tif'));
g = im2bw(f, graythresh(f));
gc = ~g;
D = bwdist(gc);
L = watershed(-D);
w = L == 0;
g2 = g & ~w;%以黑色叠加在原始二值图像上的分水岭脊线
figure
subplot(321),imshow(g)
subplot(322),imshow(gc)
subplot(323),imshow(D)
subplot(324),imshow(w)
subplot(325),imshow(g2)
%% 例10.16使用梯度和分水岭变换分割灰度图像
clc,clear,close all;
f = imread('Fig1027(a).tif'); 
h = fspecial('sobel');
fd = tofloat(f);
g = sqrt(imfilter(fd, h, 'replicate') .^ 2 +  ...
    imfilter(fd, h', 'replicate') .^ 2);
L = watershed(g);
wr = L == 0;%分水岭变换
g2 = imclose(imopen(g, ones(3,3)), ones(3,3));%闭开操作平滑梯度图像
L2 = watershed(g2);
wr2 = L2 == 0;
f2 = f;
f2(wr2) = 255;
figure
subplot(221),imshow(f)
subplot(222),imshow(g)
subplot(223),imshow(wr)
subplot(224),imshow(f2)
%% 例10.17标记符控制的分水岭分割示例
clc,clear,close all;
f = imread('Fig1028(a).tif');
h = fspecial('sobel');
fd = tofloat(f);
g = sqrt(imfilter(fd, h, 'replicate') .^ 2 + ...
    imfilter(fd, h, 'replicate') .^2);
L = watershed(g);
wr = L == 0;
rm = imregionalmin(g);%计算所有局部小区域的位置
im = imextendedmin(f, 2);%得到内部标记符集合
fim = f;
fim(im) = 175;
Lim = watershed(bwdist(im));
em = Lim == 0;
g2 = imimposemin(g, im | em);
L2 = watershed(g2);
f2 = f;
f2(L2 == 0) = 255;
figure
subplot(331),imshow(f)
subplot(332),imshow(wr)
subplot(333),imshow(rm)
subplot(334),imshow(fim)
subplot(335),imshow(em)
subplot(336),imshow(g2)
subplot(337),imshow(f2)