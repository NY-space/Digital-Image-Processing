%% ��4.1�þ��������������ָ�����ȷֲ��������
% eg������Ϊa��b�������ֲ��������  
% R = a + sqrt(b*log(1 - rand(M,N)));
%% ��4.2�ú���imnoise2����������ģʽ����)�������ݵ�ֱ��ͼ
clc,clear,close all;
r1 = imnoise2('gaussian',100000,1);
r2 = imnoise2('uniform',100000,1);
r3 = imnoise2('lognormal',100000,1);
r4 = imnoise2('rayleigh',100000,1);
r5 = imnoise2('exponential',100000,1);
r6 = imnoise2('erlang',100000,1);
bins=50
figure
subplot(231),hist(r1,bins),title('��˹�������ֱ��ͼ','FontSize',8)
subplot(232),hist(r2,bins),title('�����������ֱ��ͼ','FontSize',8)
subplot(233),hist(r3,bins),title('������̬�������ֱ��ͼ','FontSize',8)
subplot(234),hist(r4,bins),title('�����������ֱ��ͼ','FontSize',8)
subplot(235),hist(r5,bins),title('ָ���������ֱ��ͼ','FontSize',8)
subplot(236),hist(r6,bins),title('�����������ֱ��ͼ','FontSize',8)
%% ��4.3ʹ�ú���imnoise3
clc,clear,close all;
C = [0 64; 0 128; 32 32; 64 0; 128 0; -32 32];
[r, ~, S] = imnoise3(512, 512, C);
figure
subplot(321),imshow(S, [])
subplot(322),imshow(r, [])
C = [0 32; 0 64; 16 16; 32 0; 64 0; -16 16];
[r, ~, S] = imnoise3(512, 512, C);
subplot(323),imshow(S, [])
subplot(324),imshow(r, [])
C = [6 32; -2 2];
[r, ~, ~] = imnoise3(512, 512, C);
subplot(325),imshow(r, [])
A = [1 5];
[r, ~, ~] = imnoise3(512, 512, C, A);%�ϵ�Ƶ����Ǹ�Ƶ��5��
subplot(326),imshow(r, [])
%% ��4.4������������
clc,clear,close all;
f = imread('Fig0404(a).tif');
[B, c, r] = roipoly(f);%B����ģ�壬c,r����ζ�������
%B = imread('Fig0404(b).tif');
%[c, r] = size(B);
[h, npix] = histroi(f, c, r);%�������ε�ֱ��ͼ,npixΪ���������ص�
figure
subplot(221),imshow(f, [])
subplot(222),imshow(B, [])
subplot(223),bar(h, 1)
[v, unv] = statmoments(h, 2);%����ƽ��ֵ��n�����ľࣨ2��Ϊ���
v
unv
X = imnoise2('gaussian', npix, 1, 147, 20);%����147.��׼��20
subplot(224),hist(X, 130),axis([0 300 0 140])
%% ��4.5ʹ�ú���spflit��ʵ�ֿռ��˲���
clc,clear,close all;
f = imread('Fig0405(a)[without_noise].tif');
[M, N] = size(f);
R = imnoise2('salt & pepper', M, N, 0.1, 0);%������
gp = f;
gp(R == 0) = 0;
figure
subplot(321),imshow(gp, [])
R = imnoise2('salt & pepper', M, N, 0, 0.1);%������
gs = f;
gs(R == 1) = 255;
subplot(322),imshow(gs, [])
fp = spfilt(gp, 'chmean', 3, 3, 1.5);%�������˲�������ֵ������ֵ��
subplot(323),imshow(fp, [])
fs = spfilt(gs, 'chmean', 3, 3, -1.5);
subplot(324),imshow(fs, [])
fpmax = spfilt(gp, 'max', 3, 3);
subplot(325),imshow(fpmax, [])
fsmin = spfilt(gs, 'min', 3, 3);
subplot(326),imshow(fsmin, [])
%% ��4.6����Ӧ��ֵ�˲�
clc,clear,close all;
f = imread('Fig0406(a)[without_noise].tif');
g = imnoise(f, 'salt & pepper', .25);
f1 = medfilt2(g, [7 7], 'symmetric');
f2 = adpmedian(g, 7);
figure
subplot(131),imshow(g, [])
subplot(132),imshow(f1, [])
subplot(133),imshow(f2, [])
%% ��4.7ģ���ġ�������ͼ��ģ
clc,clear,close all;
f = checkerboard(8);%���ɲ��԰�
PSF = fspecial('motion', 7, 45);%ͼ���˶�ģ������
gb = imfilter(f, PSF, 'circular');
PSF
noise = imnoise2('gaussian', size(f, 1), size(f, 2), 0, sqrt(0.001));
g = gb + noise;
figure
subplot(221),imshow(pixeldup(f, 8), [])%ͼ��Ŵ�8��
subplot(222),imshow(pixeldup(gb, 8), [])
subplot(223),imshow(noise, [])
subplot(224),imshow(pixeldup(g, 8), [])%
%% ��4.8��deconvwnr�����ָ�ģ��������ͼ��
clc,clear,close all;
f = checkerboard(8);
PSF = fspecial('motion', 7, 45);
gb = imfilter(f, PSF, 'circular');
noise = imnoise2('gaussian', size(f, 1), size(f, 2), 0, sqrt(0.001));
g = gb + noise;
frest1 = deconvwnr(g , PSF);%�����Ϊ0�����˲���
Sn = abs(fft2(noise)).^2;
nA = sum(Sn(:))/numel(noise);
Sf = abs(fft2(f)).^2;
fA = sum(Sf(:))/numel(f);
R = nA/fA;%��������ȹ���
frest2 = deconvwnr(g, PSF, R);%����ά���˲���
NCORR = fftshift(real(ifft2(Sn)));
ICORR = fftshift(real(ifft2(Sf)));
frest3 = deconvwnr(g, PSF, NCORR, ICORR);%������δ�˻�ͼ�������غ���
figure
subplot(221),imshow(pixeldup(g, 8), []);
subplot(222),imshow(pixeldup(frest1, 8), []);
subplot(223),imshow(pixeldup(frest2, 8), []);
subplot(224),imshow(pixeldup(frest3, 8), []);
%% ��4.9��deconvreg������Լ������С���˷�[����]�˲�����ԭģ������ͼ��
clc,clear,close all;
f = checkerboard(8);
PSF = fspecial('motion', 7, 45);
gb = imfilter(f, PSF, 'circular');
noise = imnoise2('gaussian', size(f, 1), size(f, 2), 0, sqrt(0.001));
g = gb + noise;
frest1=deconvreg(g,PSF,4);
frest2=deconvreg(g,PSF,0.4,[1e-7 1e7]);
figure
subplot(131),imshow(g)
subplot(132),imshow(frest1)
subplot(133),imshow(frest2)
%% ��4.10���ú���deconvlucy(L-R�㷨)��ԭģ������ͼ��
clc,clear,close all;
g = checkerboard(8);
PSF = fspecial('gaussian', 7, 10);%����ɢ����
SD=0.01;
g1=imnoise(imfilter(g,PSF),'gaussian',0,SD^2);
DAMPAR=10*SD;
LIM=ceil(size(PSF,1)/2);
WEIGHT=zeros(size(g));
WEIGHT(LIM + 1:end - LIM ,LIM + 1:end - LIM)=1;
g5=deconvlucy(g,PSF,5,DAMPAR,WEIGHT);%¶��-����ɭ�㷨�������Ե���������
g10=deconvlucy(g,PSF,10,DAMPAR,WEIGHT);
g20=deconvlucy(g,PSF,20,DAMPAR,WEIGHT);
g100=deconvlucy(g,PSF,100,DAMPAR,WEIGHT);
figure
subplot(321),imshow(pixeldup(g, 8), [])
subplot(322),imshow(pixeldup(g1, 8), [])
subplot(323),imshow(pixeldup(g5, 8), [])
subplot(324),imshow(pixeldup(g10, 8), [])
subplot(325),imshow(pixeldup(g20, 8), [])
subplot(326),imshow(pixeldup(g100, 8), [])
%% ��4.11�ú���deconvblind����PSF��äȥ�����
clc,clear,close all;
PSF = fspecial('gaussian', 7, 10);
SD=0.01;
g = checkerboard(8);
g=imnoise(imfilter(g,PSF),'gaussian',0,SD^2);
INITPSF=ones(size(PSF));
DAMPAR=10*SD;
LIM=ceil(size(PSF,1)/2);
WEIGHT=zeros(size(g));
WEIGHT(LIM + 1:end - LIM ,LIM + 1:end - LIM)=1;
[g5,PSF5]=deconvblind(g,INITPSF,5,DAMPAR,WEIGHT);%�����˻�ͼ��ͨ���������PSF��ԭʼͼ��
[g10,PSF10]=deconvblind(g,INITPSF,10,DAMPAR,WEIGHT);
[g20,PSF20]=deconvblind(g,INITPSF,20,DAMPAR,WEIGHT);
figure
subplot(221),imshow(pixeldup(PSF, 73), [])
subplot(222),imshow(pixeldup(PSF5, 73), [])
subplot(223),imshow(pixeldup(PSF10, 73), [])
subplot(224),imshow(pixeldup(PSF20, 73), [])
figure
subplot(221),imshow(pixeldup(g, 73), [])
subplot(222),imshow(pixeldup(g5, 73), [])
subplot(223),imshow(pixeldup(g10, 73), [])
subplot(224),imshow(pixeldup(g20, 73), [])
%% ��4.12ʹ��radon����(����ͶӰ)
clc,clear,close all;
g1 = zeros(600, 600);
g1(100:500, 250:350) = 1;
g2 = phantom('Modified Shepp-Logan', 600);
figure
subplot(221),imshow(g1, [])
subplot(223),imshow(g2, [])
theta = 0:0.5:179.5;
[R1, xp1] = radon(g1, theta);
[R2, xp2] = radon(g2, theta);
R1 = flipud(R1');
R2 = flipud(R2');
subplot(222),imshow(R1, [], 'XData', xp1([1 end]), 'YData', [179.5 0])
axis xy
axis on
xlabel('\rho'),ylabel('\theta')
subplot(224),imshow(R2, [], 'XData', xp2([1 end]), 'YData', [179.5 0])
axis xy
axis on
xlabel('\rho'),ylabel('\theta')
%% ��4.13iradon������ʹ�ã�ͶӰ�ؽ���
clc,clear,close all;
g1 = zeros(600, 600);
g1(100:500, 250:350) = 1;
g2 = phantom('Modified Shepp-Logan', 600);
theta = 0:0.5:179.5;
R1 = radon(g1, theta);
R2 = radon(g2, theta);
f1 = iradon(R1, theta, 'none');
f2 = iradon(R2, theta, 'none');
f1_ram = iradon(R1, theta);%ʹ��Ĭ�ϵ�R-L�˲���
f2_ram = iradon(R2, theta);
f1_hamm = iradon(R1, theta, 'Hamming');%ʹ�ú����˲���
f2_hamm = iradon(R2, theta, 'Hamming');
f1_near = iradon(R1, theta, 'nearest');
f1_lin = iradon(R1, theta, 'linear');
f1_cub = iradon(R1, theta, 'cubic');
x = [400 400];
y = [400 500];
figure
subplot(241),imshow(g1, [])
subplot(242),imshow(g2, [])
subplot(243),imshow(f1, [])
subplot(244),imshow(f2, [])
subplot(245),imshow(f1_ram, [])
subplot(246),imshow(f2_ram, [])
subplot(247),imshow(f1_hamm, [])
subplot(248),imshow(f2_hamm, [])
figure
subplot(321),imshow(f1_near, [])
subplot(322),improfile(f1_near,x,y,'nearest')%�Ҷ�����ͼ
subplot(323),imshow(f1_lin, [])
subplot(324),improfile(f1_lin,x,y,'bilinear')
subplot(325),imshow(f1_cub, [])
subplot(326),improfile(f1_cub,x,y,'bicubic')
%% ��4.14ʹ�ú���fanbeam����������ͶӰ��
clc,clear,close all;
g1 = zeros(600, 600);
g1(100:500, 250:350) = 1;
g2 = phantom('Modified Shepp-Logan',600);
D = 1.5*hypot(size(g1, 1), size(g1,2))/2;
B1_line = fanbeam(g1, D, 'FanSensorGeometry','line','FanSensorSpacing',1,'FanRotationIncrement',0.5);%ֱ����������ͶӰ
B1_line = flipud(B1_line');%����ת�����·�ת��������ʾ
B2_line = fanbeam(g2, D, 'FanSensorGeometry','line','FanSensorSpacing',1,'FanRotationIncrement',0.5);
B2_line = flipud(B2_line');
%Ϊ�˱��ڱȽϣ�������ͬ��С���أ����������������Ϊ0.08��λ
B1_arc = fanbeam(g1, D, 'FanSensorGeometry','arc','FanSensorSpacing',.08,'FanRotationIncrement',0.5);%Բ��ͶӰ
B2_arc = fanbeam(g2, D, 'FanSensorGeometry','arc','FanSensorSpacing',.08,'FanRotationIncrement',0.5);
figure
subplot(221),imshow(B1_line, [])
subplot(222),imshow(B2_line, [])
subplot(223),imshow(flipud(B1_arc'), [])
subplot(224),imshow(flipud(B2_arc'), [])
%% ��4.15ʹ�ú���ifanbeam�����������ķ�ͶӰ�ؽ���
clc,clear,close all;
g = phantom('Modified Shepp-Logan', 600);
D = 1.5*hypot(size(g, 1), size(g, 2))/2;
B1 = fanbeam(g, D);%Ĭ�ϲ���
f1 = ifanbeam(B1, D);
B2 = fanbeam(g, D, 'FanRotationIncrement', 0.5, 'FanSensorSpacing', 0.5);
f2 = ifanbeam(B2, D, 'FanRotationIncrement', 0.5, 'FanSensorSpacing', 0.5, 'Filter', 'Hamming');
B3 = fanbeam(g, D, 'FanRotationIncrement', 0.5, 'FanSensorSpacing', 0.05);%���ٴ��������
f3 = ifanbeam(B3, D, 'FanRotationIncrement', 0.5, 'FanSensorSpacing', 0.05, 'Filter', 'Hamming');
figure
subplot(131),imshow(f1, [])
subplot(132),imshow(f2, [])
subplot(133),imshow(f3, [])
%% ��4.16ʹ�ú���fan2para�����κ�ƽ��ͶӰת����
clc,clear,close all;
g1 = zeros(600, 600);
g1(100:500, 250:350) = 1;
g2 = phantom('Modified Shepp-Logan',600);
D = 1.5*hypot(size(g1, 1),size(g1,2))/2;
B1_line = fanbeam(g1, D, 'FanSensorGeometry',...
    'line','FanSensorSpacing',1,...
    'FanRotationIncrement',0.5);
B2_arc = fanbeam(g2, D, 'FanSensorGeometry', 'arc',...
    'FanSensorSpacing', .08, 'FanRotationIncrement', 0.5);
P1_line = fan2para(B1_line,D,'FanRotationIncrement',0.5,...
    'FanSensorGeometry','line',...
    'FanSensorSpacing',1,...
    'ParallelCoverage','halfcycle',...
    'ParallelRotationIncrement', 0.5,...
    'ParallelSensorSpacing',1);
P2_arc = fan2para(B2_arc, D, 'FanRotationIncrement',0.5,...
    'FanSensorGeometry','arc',...
    'FanSensorSpacing',0.08,...
    'ParallelCoverage','halfcycle',...
    'ParallelRotationIncrement', 0.5,...
    'ParallelSensorSpacing',1);
P1_line = flipud(P1_line');
P2_arc = flipud(P2_arc');
figure
subplot(121),imshow(P1_line, [])
subplot(122),imshow(P2_arc, [])





