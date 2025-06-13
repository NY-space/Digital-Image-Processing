%% 例8.1计算熵(编码最小平均字长)
clc,clear,close all;
f = [119 123 168 119; 123 119 168 168];
f= [f; 119 119 107 119; 107 107 119 119];
p = hist(f(:), 8);
p = p / sum(p)
h = ntrop(f)

%% 例8.2MATLAB中的变长编码映射
clc,clear,close all;
f2 = uint8([2 3 4 2; 3 2 4 4; 2 2 1 2; 1 1 2 2])
whos('f2')
c = huffman(hist(double(f2(:)), 4))
h1f2 = c(f2(:))'
whos('h1f2')
h2f2 = char(h1f2)'
whos('h2f2')
h2f2 = h2f2(:);
h2f2(h2f2 == ' ') = [];%消除空格
whos('h2f2')
h3f2 = mat2huff(f2)
whos('h3f2')
hcode = h3f2.code;
whos('hcode')
dec2bin(double(hcode))
%% 例8.3用mat2huff进行编码
clc,clear,close all;
f = imread('Fig0804(a).tif');
c = mat2huff(f);
cr1 = imratio(f, c)%计算图像中字节的比例  %压缩为原来的80%
save SqueezeTracy c;
cr2 = imratio(f, 'SqueezeTracy.mat')%matlab数据文件的开销
%% 例8.4使用huff2mat进行解码
clc,clear,close all;
load SqueezeTracy;
g = huff2mat(c);
f = imread('Fig0804(a).tif');
rmse = compare(f, g)%原始图像和解压后的图像之间的均方根误差为0
%% 例8.5无损预测编码
clc,clear,close all;
f = imread('Fig0807(c).tif');
ntrop(f)
e = mat2lpc(f);%预测编码
figure
subplot(121),imshow(mat2gray(e));
ntrop(e)
c = mat2huff(e);
cr = imratio(f, c)
[h, x] = hist(e(:) * 512, 512);
subplot(122),bar(x, h, 'k');
g = lpc2mat(huff2mat(c));%解码处理
compare(f, g)
%% 例8.6量化压缩（有损压缩）
clc,clear,close all;
f = imread('Fig0810(a).tif');
q1 = quantize(f, 16);%均量化16级（伪轮廓）
q2 = quantize(f, 16, 'igs');%IGS量化16级，对每个像素增加伪随机数（改进的灰度级量化）
figure
subplot(131),imshow(f)
subplot(132),imshow(q1)
subplot(133),imshow(q2)
%% 例8.7结合IGS（改进的灰度级量化）量化的无损预测和霍华夫编码
clc,clear,close all;
f = imread('Fig0810(a).tif');
q = quantize(f, 4, 'igs');
qs = double(q) / 16;
e = mat2lpc(qs);
c = mat2huff(e);
imratio(f, c)
ne = huff2mat(c);
nqs = lpc2mat(ne);
nq = 16 * nqs;
compare(q, nq)
compare(f, nq)%解压图像的均方根误差大约是7个灰度级，误差源于量化步骤
%% 例8.8JPEG压缩(基于离散余弦变换)
clc,clear,close all;
f = imread('Fig0804(a).tif');
c1 = im2jpeg(f);
f1 = jpeg2im(c1);
imratio(f,c1)
compare(f,f1,3)
c4 = im2jpeg(f, 4);%4乘以归一化数组
f4 = jpeg2im(c4);
imratio(f, c4)
compare(f,f4,3)
%% 例8.9JPEG 2000压缩
clc,clear,close all;
f = imread('Fig0804(a).tif');
c1 = im2jpeg2k(f,5,[8 8.5]);
f1 = jpeg2k2im(c1);
rms1 = compare(f,f1)
cr1 = imratio(f,c1)
c2 = im2jpeg2k(f,5,[8 7]);
f2 = jpeg2k2im(c2);
rms2 = compare(f,f2)
cr2 = imratio(f,c2)
c3 = im2jpeg2k(f,1,[1 1 1 1]);
f3 = jpeg2k2im(c3);
rms3 = compare(f,f3)
cr3 = imratio(f,c3)
%% 例8.10时间冗余(时间上彼此接近的像素相关)霍夫曼编码和线性预测编码移除
clc,clear,close all;
f2 = imread('shuttle.tif', 2);
ntrop(f2)
e2 = mat2lpc(f2);
ntrop(e2,512)
c2 = mat2huff(e2);
imratio(f2,c2)
f1 = imread('shuttle.tif', 1);
ne2 = double(f2) - double(f1);
ntrop(ne2,512)
nc2 = mat2huff(ne2);
imratio(f2,nc2)
%% 例8.11运动补偿视频压缩（报错)
clc,clear,close all;
cv = tifs2cv('shuttle.tif', 16, [8 8]);
imratio('shuttle.tif',cv)
figure
subplot(131),showmo(cv,2);
tic; cv2 = tifs2cv('shuttle.tif', 16, [8 8], 1); toc
%tic; cv2tifs(cv2, 'ss2.tif'); toc
imratio('shuttle.tif',cv2)
%compare(imread('shuttle.tif',16),imread('ss2.tif',8))
%compare(imread('shuttle.tif',16), imread('ss2.tif',16))

