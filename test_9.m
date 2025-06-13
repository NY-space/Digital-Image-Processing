%% 例9.1膨胀的应用
clc,clear,close all;
A=imread('Fig0906(a).tif');
B=[0 1 0;1 1 1;0 1 0];%结构元
D=imdilate(A,B);%膨胀
figure
subplot(121),imshow(A)
subplot(122),imshow(D)
%% 例9.2用strel分解结构元
clc,clear,close all;
se=strel('diamond',5)
decomp=getsequence(se);%提取并检查分解中的单独的结构元
whos%列出工作区中的变量及大小和类型
decomp(1)%索引分解后的4个结构元
decomp(2)
decomp(3)
decomp(4)
%% 例9.3腐蚀的说明
clc,clear,close all;
A=imread('Fig0908(a).tif');
se=strel('disk',15);
E15=imerode(A,se);
figure
subplot(221),imshow(A);
subplot(222),imshow(E15);
se=strel('disk',5);
E5=imerode(A,se);
subplot(223),imshow(E5);
se=strel('disk',40);
E40=imerode(A,se);
subplot(224),imshow(E40);
%% 例9.4使用imopen和imclose（开操作，闭操作）
clc,clear,close all;
f=imread('Fig0910(a).tif');
se=strel('square',40);
fo=imopen(f,se);
fc=imclose(f,se);
foc=imclose(fo,se);
figure
subplot(221),imshow(f),title('原始图像');
subplot(222),imshow(fo),title('开操作');
subplot(223),imshow(fc),title('闭操作');
subplot(224),imshow(foc),title('先开后闭');
%噪声消除
f=imread('Fig0911(a).tif');
se=strel('square',6);
fo=imopen(f,se);
foc=imclose(fo,se);
figure
subplot(131),imshow(f),title('原始图像');
subplot(132),imshow(fo),title('开操作');
subplot(133),imshow(foc),title('先开后闭');
%% 例9.5使用函数bwhitmiss（击中或击不中变换）
clc,clear,close all;
f=imread('Fig0913(a).tif');
B1=strel([0 0 0 ;0 1 1;0 1 0]);
B2=strel([1 1 1;1 0 0;1 0 0]);
g=bwhitmiss(f,B1,B2);
%间隔矩阵
interval=[-1 -1 -1;-1 1 1;-1 1 0];
g=bwhitmiss(f,B1,B2);
figure
subplot(121),imshow(f);
subplot(122),imshow(g);
%% 例9.6用二值图像及基于查找表的计算玩Conway的Game of Life生命游戏
clc,clear,close all;
lut=makelut(@conwaylaws,3);
bw1=[0 0 0 0 0 0 0 0 0 0 
    0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 1 0 0 0
    0 0 0 1 1 1 1 0 0 0
    0 0 1 0 0 0 0 1 0 0
    0 0 1 0 0 0 0 1 0 0
    0 0 1 0 0 0 0 1 0 0
    0 0 0 1 1 1 1 0 0 0
    0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0
    ]
imshow(bw1,'InitialMagnification','fit'),title('Generation 1')
bw2=applylut(bw1,lut);
figure,imshow(bw2,'InitialMagnification','fit'),title('Generation 2')
bw3=applylut(bw2,lut);
figure,imshow(bw3,'InitialMagnification','fit'),title('Generation 3')
%% 例9.7计算和显示连通分量的质心
clc,clear,close all;
f=imread('Fig0917(a).tif');
[L,n]=bwlabel(f);
[r,c]=find(L==3);
rbar=mean(r);
cbar=mean(c);
imshow(f)
hold on
for k=1:n
    [r,c]=find(L==k);
    rbar=mean(r);
    cbar=mean(c);
    plot(cbar,rbar,'Marker','o','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',10);
    plot(cbar,rbar,'Marker','*','MarkerEdgeColor','w');
end
%% 例9.8通过重建进行开操作
clc,clear,close all;
f=imread('Fig0922(a).tif');
fe=imerode(f,ones(51,1));%用竖线腐蚀
fo=imopen(f,ones(51,1));%用竖线进行开操作
fobr=imreconstruct(fe,f);%用竖线通过重建进行的开操作
figure
subplot(221),imshow(f)
subplot(222),imshow(fe)
subplot(223),imshow(fo)
subplot(224),imshow(fobr)
%% 例9.9用开操作和闭操作做形态学平滑
clc,clear,close all;
f=imread('Fig0925(a).tif');
se=strel('disk',5);
fo=imopen(f,se);
foc=imclose(fo,se);
fc=imclose(f,se);
fco=imopen(fc,se);
figure
subplot(321),imshow(f);
subplot(322),imshow(fo);
subplot(323),imshow(foc);
subplot(324),imshow(fc);
subplot(325),imshow(fco);

fasf=f;
for k=2:5
    se=strel('disk',k);
    fasf=imclose(imopen(fasf,se),se);
end
subplot(326),imshow(fasf);%交替顺序滤波
%% 例9.10非均匀背景的补偿
clc,clear,close all;
f=imread('Fig0926(a).tif');
se=strel('disk',10);
fo=imopen(f,se);%补偿非均匀照明背景
f1=f-fo;
f2=imtophat(f,se);
figure
subplot(141),imshow(f);
subplot(142),imshow(fo);
subplot(143),imshow(f1);
subplot(144),imshow(f2);
%% 例9.11粒度测定
clc,clear,close all;
f=imread('Fig0925(a).tif');
umpixels=zeros(1,36);
for k=0:35
    se=strel('disk',k);
    fo=imopen(f,se);
    sumpixels(k+1)=sum(fo(:));
end
figure
subplot(121),plot(0:35,sumpixels),xlabel('k'),ylabel('Surface area')%相对于结构元半径的表面区域
subplot(122),plot(-diff(sumpixels)),xlabel('k'),ylabel('Surface area reduction')%相对于结构元半径的表面区域减少
%% 例9.12用重建移去复杂的背景操作
clc,clear,close all;
f=imread('Fig0930(a).tif');
f_obr=imreconstruct(imerode(f,ones(1,71)),f);
f_o=imopen(f,ones(1,71));
f_thr=f-f_obr;
f_th=f-f_o;
g_obr=imreconstruct(imerode(f_thr,ones(1,11)),f_thr);
g_obrd=imdilate(g_obr,ones(1,21));
f2=imreconstruct(min(g_obrd,f_thr),f_thr);
figure
subplot(331),imshow(f),title('原始图像');
subplot(332),imshow(f_obr),title('重建开操作');
subplot(333),imshow(f_o),title('开操作');
subplot(334),imshow(f_thr),title('重建的顶帽操作');
subplot(335),imshow(f_th),title('顶帽操作');
subplot(336),imshow(g_obr),title('使用水平线针对（4）进行重建的开操作');
subplot(337),imshow(g_obrd),title('用水平线对（6）进行的膨胀处理');
subplot(338),imshow(f2),title('最后的重建处理');
















;