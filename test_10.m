%% ��10.1����
clc,clear,close all;
f = imread('Fig1002(a).tif');
w = [-1 -1 -1; -1 8 -1; -1 -1 -1];
g = abs(imfilter(tofloat(f),w));
T = max(g(:));
g = g >= T;
figure
subplot(121),imshow(f)
subplot(122),imshow(g)
%% ��10.2�߼��
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
%% ��10.3Sobel��Ե������ӵ�ʹ��
clc,clear,close all;
f = imread('Fig1006(a).tif');
[gv, t] = edge(f, 'sobel', 'vertical');%��ֱ��Ե���
figure
subplot(321),imshow(f)
subplot(322),imshow(gv)
t
gv =edge(f, 'sobel', 0.15, 'vertical');%�趨��ֵ�Ĵ�ֱ��Ե���
subplot(323),imshow(gv)
gboth = edge(f, 'sobel', 0.15);%�趨��ֵ�Ĵ�ֱ��ˮƽ��Ե���
subplot(324),imshow(gboth)
wneg45 = [-2 -1 0; -1 0 1; 0 1 2]%-45���Ե���
gneg45 = imfilter(tofloat(f), wneg45, 'replicate');
T = 0.3*max(abs(gneg45(:)));
gneg45 = gneg45 >= T;
subplot(325),imshow(gneg45)
wpos45 = [0 1 2; -1 0 1; -2 -1 0]%+45���Ե���
gpos45 = imfilter(tofloat(f), wpos45, 'replicate');
T = 0.3*max(abs(gpos45(:)));
gpos45 = gpos45 >= T;
subplot(326),imshow(gpos45)
%% ��10.4Sobel,LoG��Canny��Ե������ӵıȽ�
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
subplot(321),imshow(gSobel_default)%����Ĭ��ֵ���������ò���
subplot(322),imshow(gSobel_best)
subplot(323),imshow(gLoG_default)
subplot(324),imshow(gLoG_best)
subplot(325),imshow(gCanny_default)
subplot(326),imshow(gCanny_best)
%% ��10.5����任��˵��
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
%% ��10.6�û���任����������
clc,clear,close all;
f = imread('Fig1006(a).tif');
f = tofloat(f);
f = edge(f, 'canny', [0.04 0.10], 1.5);
[H, theta, rho] = hough(f, 'ThetaResolution', 0.2);
figure
subplot(121),imshow(H, [], 'XData',theta,'YData',rho,'InitialMagnification','fit')
axis on,axis normal
xlabel('\theta'),ylabel('\rho')
peaks = houghpeaks(H, 5);%����任�ķ�ֵ
hold on
plot(theta(peaks(:,2)),rho(peaks(:,1)),...
    'linestyle', 'none', 'marker', 's', 'color', 'w')
lines = houghlines(f, theta, rho, peaks);
subplot(122),imshow(f),hold on
for k = 1:length(lines)
    xy = [lines(k).point1 ; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color',[.8 .8 .8]);
end
%% ��10.7����ȫ����ֵ
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
%% ��10.8��ʹ��Otsu's������10.3.2�ڵĻ���ȫ����ֵ�������ָ�ͼ��ıȽ�
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
%% ����ֵ����֮ǰ�ȶ�ͼ�����ƽ��
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
%% ��10.9ʹ�û����ݶȵı�Ե��Ϣ�Ľ�ȫ����ֵ����
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
%% ��10.10��������˹��Ե��Ϣ�Ľ�ȫ����ֵ����
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
Q = percentile2i(h, 0.995);%����Ҷȼ�
markerImage = lap > Q;
fp = f.*markerImage;
subplot(234),imshow(fp)
hp = imhist(fp);
hp(1) = 0;
subplot(235),bar(hp)
T = otsuthresh(hp);
g = im2bw(f,T);
subplot(236),imshow(g)
%% ��10.11��ȫ�ֺ;ֲ���ֵ����ıȽ�
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
%% ��10.12�����ƶ�ƽ����ͼ����ֵ����
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
%% ��10.13ʹ������������⺸�ӿ�϶
clc,clear,close all;
f = imread('Fig1020(a).tif');
[g, NR, SI, TI] = regiongrow(f, 1, 0.26);
figure
subplot(221),imshow(f)
subplot(222),imshow(SI)%���ӵ�
subplot(223),imshow(TI)%��ʾ����ͨ����ֵ���Եĵ�
subplot(224),imshow(g)
figure
imhist(f)
ylim('auto')
%% ��10.14ʹ�����������ͺϲ���ͼ��ָ�
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
%% ��10.15ʹ�þ���任�ͷ�ˮ��任�ָ��ֵͼ��
clc,clear,close all;
f= tofloat(~imread('Fig1026(a).tif'));
g = im2bw(f, graythresh(f));
gc = ~g;
D = bwdist(gc);
L = watershed(-D);
w = L == 0;
g2 = g & ~w;%�Ժ�ɫ������ԭʼ��ֵͼ���ϵķ�ˮ�뼹��
figure
subplot(321),imshow(g)
subplot(322),imshow(gc)
subplot(323),imshow(D)
subplot(324),imshow(w)
subplot(325),imshow(g2)
%% ��10.16ʹ���ݶȺͷ�ˮ��任�ָ�Ҷ�ͼ��
clc,clear,close all;
f = imread('Fig1027(a).tif'); 
h = fspecial('sobel');
fd = tofloat(f);
g = sqrt(imfilter(fd, h, 'replicate') .^ 2 +  ...
    imfilter(fd, h', 'replicate') .^ 2);
L = watershed(g);
wr = L == 0;%��ˮ��任
g2 = imclose(imopen(g, ones(3,3)), ones(3,3));%�տ�����ƽ���ݶ�ͼ��
L2 = watershed(g2);
wr2 = L2 == 0;
f2 = f;
f2(wr2) = 255;
figure
subplot(221),imshow(f)
subplot(222),imshow(g)
subplot(223),imshow(wr)
subplot(224),imshow(f2)
%% ��10.17��Ƿ����Ƶķ�ˮ��ָ�ʾ��
clc,clear,close all;
f = imread('Fig1028(a).tif');
h = fspecial('sobel');
fd = tofloat(f);
g = sqrt(imfilter(fd, h, 'replicate') .^ 2 + ...
    imfilter(fd, h, 'replicate') .^2);
L = watershed(g);
wr = L == 0;
rm = imregionalmin(g);%�������оֲ�С�����λ��
im = imextendedmin(f, 2);%�õ��ڲ���Ƿ�����
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