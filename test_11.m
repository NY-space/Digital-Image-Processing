%% 例11.1函数bwboundarirs和bound2im的使用
clc,clear,close all;
f = [
 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
 0 1 1 1 1 1 1 1 1 1 1 1 1 0;
 0 1 1 1 1 1 1 1 1 1 1 1 1 0;
 0 1 1 0 0 0 0 0 0 0 0 1 1 0;
 0 1 1 0 1 1 1 1 1 1 0 1 1 0;
 0 1 1 0 1 1 1 1 1 1 0 1 1 0;
 0 1 1 0 1 1 0 0 1 1 0 1 1 0;
 0 1 1 0 1 1 0 0 1 1 0 1 1 0;
 0 1 1 0 1 1 1 1 1 1 0 1 1 0;
 0 1 1 0 1 1 1 1 1 1 0 1 1 0;
 0 1 1 0 0 0 0 0 0 0 0 1 1 0;
 0 1 1 1 1 1 1 1 1 1 1 1 1 0;
 0 1 1 1 1 1 1 1 1 1 1 1 1 0;
 0 0 0 0 0 0 0 0 0 0 0 0 0 0
];
B = bwboundaries(f, 'noholes');%8连通仅提取区域的边缘
numel(B)
b = cat(1, B{:});
[M, N] = size(f);
imgae = bound2im(b, M, N)
[B, L, NR, A] = bwboundaries(f);
numel(B)
numel(B) - NR
bR = cat(1, B{1:2}, B{4});
imageBoundaries = bound2im(bR, M, N);
imageNumveredBoundaries = imageBoundaries.*L
bR = cat(1, B{:});
imageBoundaries = bound2im(bR, M, N);
imageNumberedBoundaries = imageBoundaries.*L
find(A(:,1))
find(A(1,:))
A
full(A)
%% 例11.2佛雷曼链码及其某些变体
clc,clear,close all;
f = imread('Fig1103(a).tif');
figure
subplot(231),imshow(f)
h = fspecial('average', 9);
g = imfilter(f, h, 'replicate');
subplot(232),imshow(g)
gB = im2bw(g, 0.5);
subplot(233),imshow(gB)
B = bwboundaries(gB, 'noholes');
d = cellfun('length', B);
[maxd, k] = max(d);
b = B{k};
[M N] = size(g);
g = bound2im(b, M, N);
subplot(234),imshow(g)
[s, su] = bsubsamp(b, 50);
g2 = bound2im(s, M, N);
subplot(235),imshow(g2)
cn = connectpoly(s(:, 1), s(:, 2));
g3 = bound2im(cn, M, N);
subplot(236),imshow(g3)
c = fchcode(su);
c.x0y0
c.fcc
c.mm
c.diff
c.diffmm
%% 例11.3得到由细胞组合体包围的区域边界（原始图像难易绘制）
%% 例11.4使用函数im2minperpoly（-m函数报错)
clc,clear,close all;
f = imread('Fig1107(a).tif');
figure
subplot(321),imshow(f)
B = bwboundaries(f, 4, 'noholes');
b = B{1};
[M, N] = size(f);
bOriginal = bound2im(b, M, N);
subplot(322),imshow(bOriginal)

[X, Y] = im2minperpoly(f, 2);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(323),imshow(bCellsize2)

[X, Y] = im2minperpoly(f, 3);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(324),imshow(bCellsize2)

[X, Y] = im2minperpoly(f, 4);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(325),imshow(bCellsize2)

[X, Y] = im2minperpoly(f, 8);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(326),imshow(bCellsize2)

figure
[X, Y] = im2minperpoly(f, 10);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(221),imshow(bCellsize2)

[X, Y] = im2minperpoly(f, 16);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(222),imshow(bCellsize2)

[X, Y] = im2minperpoly(f, 20);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(223),imshow(bCellsize2)

[X, Y] = im2minperpoly(f, 32);
b2 = connectpoly(X, Y);
bCellsize2 = bound2im(b2, M, N);
subplot(224),imshow(bCellsize2)
%% 例11.5标记
clc,clear,close all;
fsq = imread('Fig1111(a).tif');
ftr = imread('Fig1111(b).tif');
figure
subplot(221),imshow(fsq)
subplot(222),imshow(ftr)
bSq = bwboundaries(fsq, 'noholes');
[distSq, angleSq] = signature(bSq{1});
subplot(223),plot(angleSq, distSq)
bSq = bwboundaries(ftr, 'noholes');
[distSq, angleSq] = signature(bSq{1});
subplot(224),plot(angleSq, distSq)
%% 例11.6计算区域的骨骼
clc,clear,close all;
f = imread('Fig1113(a).tif');
figure
subplot(231),imshow(f)
h = fspecial('gaussian', 25, 15);
g = tofloat(imfilter(f, h, 'replicate'));%平滑处理
subplot(232),imshow(g)
g = im2bw(g, 1.5*graythresh(g));%阈值处理
subplot(233),imshow(g)
s = bwmorph(g, 'skel', Inf);
subplot(234),imshow(s)
s1 = bwmorph(s, 'spur', 8);%刺状突起去除8次后的骨骼
subplot(235),imshow(s1)
s2 = bwmorph(s, 'spur', 7);
subplot(236),imshow(s2)
%% 11.7傅里叶描述子
clc,clear,close all;
f = imread('Fig1116(a).tif');
figure
subplot(121),imshow(f)
b = bwboundaries(f, 'noholes');
b = b{1};
bim = bound2im(b, size(f, 1), size(f, 2));
subplot(122),imshow(bim)%边界点1090
z = frdescp(b);
s546 = ifrdescp(z, 546);%调整傅里叶描述子
s546im = bound2im(s546, size(f, 1), size(f, 2));
s110 = ifrdescp(z, 110);
s110im = bound2im(s110, size(f, 1), size(f, 2));
s56 = ifrdescp(z, 56);
s56im = bound2im(s56, size(f, 1), size(f, 2));
s28 = ifrdescp(z, 28);
s28im = bound2im(s28, size(f, 1), size(f, 2));
s14 = ifrdescp(z, 14);
s14im = bound2im(s14, size(f, 1), size(f, 2));
s8 = ifrdescp(z, 8);
s8im = bound2im(s8, size(f, 1), size(f, 2));
figure
subplot(231),imshow(s546im)
subplot(232),imshow(s110im)
subplot(233),imshow(s56im)
subplot(234),imshow(s28im)
subplot(235),imshow(s14im)
subplot(236),imshow(s8im)
%% 例11.8在灰度图中使函数cornermetric和conrnerprocess寻找拐角
clc,clear,close all;
f = imread('Fig1119(a).tif');
figure
subplot(321),imshow(f)
CH = cornermetric(f, 'Harris');
CH(CH < 0) = 0;
CH = mat2gray(CH);
subplot(323),imshow(imcomplement(CH))
CM = cornermetric(f, 'MinimumEigenvalue');
CM = mat2gray(CM);
subplot(324),imshow(imcomplement(CM))
hH = imhist(CH);
hM = imhist(CM);
TH = percentile2i(hH, 0.9945);
TM = percentile2i(hM, 0.9970);
cpH = cornerprocess(CH, TH, 1);
cpM = cornerprocess(CM, TM, 1);
subplot(325),imshow(cpH)
subplot(326),imshow(cpM)

[xH yH] = find(cpH);
figure, subplot(221),imshow(f)
hold on
plot(yH(:)', xH(:)', 'wo')
[xM yM] = find(cpM);
subplot(222),imshow(f)
hold on
plot(yM(:)', xM(:)', 'wo')

cpH = cornerprocess(CH, TH, 5);
cpM = cornerprocess(CM, TM, 5);
[xH yH] = find(cpH);
subplot(223),imshow(f)
hold on
plot(yH(:)', xH(:)', 'wo')
[xM yM] = find(cpM);
subplot(224),imshow(f)
hold on
plot(yM(:)', xM(:)', 'wo')
%% 例11.9函数regionprops的运用
clc,clear,close all;
B = imread('Fig1119(a).tif');
B = bwlabel(B);
D = regionprops(B, 'area', 'boundingbox');
A = [D.Area];%区域面积
NR = numel(A);%区域个数
V = cat(1, D.BoundingBox);
%% 例11.10统计纹理的度量
clc,clear,close all;
f1 = imread('Fig1121(a).tif');
f2 = imread('Fig1121(b).tif');
f3 = imread('Fig1121(c).tif');
figure
subplot(231),imshow(f1)%光滑纹理
subplot(232),imshow(f2)%粗糙纹理
subplot(233),imshow(f3)%周期纹理
f1 = imhist(f1);
f2 = imhist(f2);
f3 = imhist(f3);
subplot(234),plot(f1)
subplot(235),plot(f2)
subplot(236),plot(f3)
t = statxture(f1)
%% 例11.11基于共生矩阵的描述子
clc,clear,close all;
f2 = imread('Fig1124(b).tif');
G2 = graycomatrix(f2, 'NumLevels', 256);
G2n = G2/sum(G2(:));
stats2 = graycoprops(G2, 'all');
maxProbability2 = max(G2n(:));
contrast2 = stats2.Contrast;
corr2 = stats2.Correlation;
energy2 = stats2.Homogeneity;
hom2 = stats2.Homogeneity;
for I = 1:size(G2n, 1);
    sumcols(I) = sum(-G2n(I, 1:end).*log2(G2n(I, 1:end)...
        +eps));
end
entropy2 = sum(sumcols);
offsets = [zeros(50,1) (1:50)'];
G2 = graycomatrix(f2,'Offset',offsets);
stats2 = graycoprops(G2, 'Correlation');
figure
subplot(132),plot([stats2.Correlation]);
xlabel('Horizontal Offset')
ylabel('Correlation')

f1 = imread('Fig1124(a).tif');
G2 = graycomatrix(f1, 'NumLevels', 256);
G2n = G2/sum(G2(:));
stats2 = graycoprops(G2, 'all');
maxProbability2 = max(G2n(:));
contrast2 = stats2.Contrast;
corr2 = stats2.Correlation;
energy2 = stats2.Homogeneity;
hom2 = stats2.Homogeneity;
for I = 1:size(G2n, 1);
    sumcols(I) = sum(-G2n(I, 1:end).*log2(G2n(I, 1:end)...
        +eps));
end
entropy2 = sum(sumcols);
offsets = [zeros(50,1) (1:50)'];
G2 = graycomatrix(f1,'Offset',offsets);
stats2 = graycoprops(G2, 'Correlation');
subplot(131),plot([stats2.Correlation]);
xlabel('Horizontal Offset')
ylabel('Correlation')

f3 = imread('Fig1124(c).tif');
G2 = graycomatrix(f3, 'NumLevels', 256);
G2n = G2/sum(G2(:));
stats2 = graycoprops(G2, 'all');
maxProbability2 = max(G2n(:));
contrast2 = stats2.Contrast;
corr2 = stats2.Correlation;
energy2 = stats2.Homogeneity;
hom2 = stats2.Homogeneity;
for I = 1:size(G2n, 1);
    sumcols(I) = sum(-G2n(I, 1:end).*log2(G2n(I, 1:end)...
        +eps));
end
entropy2 = sum(sumcols);
offsets = [zeros(50,1) (1:50)'];
G2 = graycomatrix(f3,'Offset',offsets);
stats2 = graycoprops(G2, 'Correlation');
subplot(133),plot([stats2.Correlation]);
xlabel('Horizontal Offset')
ylabel('Correlation')

figure
subplot(311),imshow(f1)
subplot(312),imshow(f2)
subplot(313),imshow(f3)
%% 例11.12计算谱纹理
clc,clear,close all;
f1 = imread('Fig1126(a).tif');
f2 = imread('Fig1126(b).tif');
f1_fft = fftshift(fft2(f1));
f2_fft = fftshift(fft2(f2));
figure
subplot(221),imshow(f1)
subplot(222),imshow(f2)
subplot(223),imshow(im2uint8(mat2gray(log(1+double(abs(f1_fft))))),[])
subplot(224),imshow(im2uint8(mat2gray(log(1+double(abs(f2_fft))))),[])
[srad, sang, S] = specxture(f1);
figure
subplot(221),plot(srad)
subplot(222),plot(sang)
[srad, sang, S] = specxture(f2);
subplot(223),plot(srad)
subplot(224),plot(sang)
%% 例11.13不变矩
clc,clear,close all;
f = imread('Fig1128(a)[original].tif');
fp = padarray(f, [84 84], 'both');
figure
subplot(231),imshow(fp)
ftrans = zeros(568, 568, 'uint8');
ftrans(151:550,151:550) = f;
subplot(232),imshow(ftrans)
fhs = f(1:2:end, 1:2:end);
fhsp = padarray(fhs, [184 184], 'both');
subplot(233),imshow(fhsp)
fm = fliplr(f);
fmp = padarray(fm, [84 84], 'both');
subplot(234),imshow(fmp)
fr45 = imrotate(f, 45, 'bilinear');
fr90 = imrotate(f, 90, 'bilinear');
fr90p = padarray(fr90, [84 84], 'both');
subplot(235),imshow(fr45)
subplot(236),imshow(fr90p)
phi = invmoments(f);
format short e
phi
format short
phinorm = -sign(phi).*(log10(abs(phi)))
%% 例11.14主分量的使用
clc,clear,close all;
f1 = imread('Fig1130(a).tif');
f2 = imread('Fig1130(b).tif');
f3 = imread('Fig1130(c).tif');
f4 = imread('Fig1130(d).tif');
f5 = imread('Fig1130(e).tif');
f6 = imread('Fig1130(f).tif');
figure
subplot(321),imshow(f1)
subplot(322),imshow(f2)
subplot(323),imshow(f3)
subplot(324),imshow(f4)
subplot(325),imshow(f5)
subplot(326),imshow(f6)

S = cat(3,f1,f2,f3,f4,f5,f6);
X = imstack2vectors(S);
P = principalcomps(X,6);
g1 = P.Y(:,1);
g1 = reshape(g1,512,512);
g2 = P.Y(:,2);
g2 = reshape(g2,512,512);
g3 = P.Y(:,3);
g3 = reshape(g3,512,512);
g4 = P.Y(:,4);
g4 = reshape(g4,512,512);
g5 = P.Y(:,5);
g5 = reshape(g5,512,512);
g6 = P.Y(:,6);
g6 = reshape(g6,512,512);
figure
subplot(321),imshow(g1,[])
subplot(322),imshow(g2,[])
subplot(323),imshow(g3,[])
subplot(324),imshow(g4,[])
subplot(325),imshow(g5,[])
subplot(326),imshow(g6,[])
d = diag(P.Cy);
P = principalcomps(X,2);
h1 = P.X(:,1);
h1 = mat2gray(reshape(h1,512,512));
D1 = tofloat(f1) - h1;

h2 = P.X(:,2);
h2 = mat2gray(reshape(h2,512,512));
D2 = tofloat(f2) - h2;

h3 = P.X(:,3);
h3 = mat2gray(reshape(h3,512,512));
D3 = tofloat(f2) - h3;

h4 = P.X(:,4);
h4 = mat2gray(reshape(h4,512,512));
D4 = tofloat(f4) - h4;

h5 = P.X(:,5);
h5 = mat2gray(reshape(h5,512,512));
D5 = tofloat(f5) - h5;

h6 = P.X(:,6);
h6 = mat2gray(reshape(h6,512,512));
D6 = tofloat(f6) - h6;
figure
subplot(321),imshow(D1,[])
subplot(322),imshow(D2,[])
subplot(323),imshow(D3,[])
subplot(324),imshow(D4,[])
subplot(325),imshow(D5,[])
subplot(326),imshow(D6,[])
P.ems
figure
subplot(121),imshow(abs(D1 - tofloat(f1)))
subplot(122),imshow(abs(D6 - tofloat(f6)))
%% 例11.15用主分量调整物体
clc,clear,close all;
f = im2bw(imread('Fig1134(a).tif'));
[x1 x2] = find(f);%提取1值坐标
X = [x1 x2];
P = principalcomps(X, 2);
A = P.A;
Y = (A*(X'))';
miny1 = min(Y(:,1));
miny2 = min(Y(:,2));
y1 = round(Y(:,1) - miny1 + min(x1));
y2 = round(Y(:,2) - miny2 + min(x2));
idx = sub2ind(size(f),y1,y2);
fout = false(size(f));
fout(idx) = 1;
fout = imclose(fout, ones(3));
fout = rot90(fout,2);
figure
subplot(231),imshow(f)
subplot(234),imshow(fout)

f = im2bw(imread('Fig1134(b).tif'));
[x1 x2] = find(f);
X = [x1 x2];
P = principalcomps(X, 2);
A = P.A;
Y = (A*(X'))';
miny1 = min(Y(:,1));
miny2 = min(Y(:,2));
y1 = round(Y(:,1) - miny1 + min(x1));
y2 = round(Y(:,2) - miny2 + min(x2));
idx = sub2ind(size(f),y1,y2);
fout = false(size(f));
fout(idx) = 1;
fout = imclose(fout, ones(3));
fout = rot90(fout,2);
subplot(232),imshow(f)
subplot(235),imshow(fout)

f = im2bw(imread('Fig1134(c).tif'));
[x1 x2] = find(f);
X = [x1 x2];
P = principalcomps(X, 2);
A = P.A;
Y = (A*(X'))';
miny1 = min(Y(:,1));
miny2 = min(Y(:,2));
y1 = round(Y(:,1) - miny1 + min(x1));
y2 = round(Y(:,2) - miny2 + min(x2));
idx = sub2ind(size(f),y1,y2);
fout = false(size(f));
fout(idx) = 1;
fout = imclose(fout, ones(3));
fout = rot90(fout,2);
subplot(233),imshow(f)
subplot(236),imshow(fout)
