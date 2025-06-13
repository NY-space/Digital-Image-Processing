%% 例5.1 创建自定义的tform结构，并用来变换点
% 变换案例1，因数3水平的放大，因数2垂直的放大
clc,clear,close all;
forward_fcn=@(wz,tdata)[3*wz(:,1),2*wz(:,2)];%正向函数
inverse_fcn=@(xy,tdata)[xy(:,1)/3,xy(:,2)/2];%反向函数
tform1=maketform('custom',2,2,forward_fcn,inverse_fcn,[])%创建空间变换结构
WZ=[1 1;3 2];
XY=tformfwd(WZ,tform1)
WZ2=tforminv(XY,tform1)
%变换案例2，垂直坐标因子移动水平坐标，并保持垂直坐标不变
forward_fcn=@(wz,tdata)[wz(:,1)+0.4*wz(:,2),wz(:,2)];%正向函数
inverse_fcn=@(xy,tdata)[xy(:,1)-0.4*xy(:,2),xy(:,2)];%反向函数
tform2=maketform('custom',2,2,forward_fcn,inverse_fcn,[])%创建空间变换结构
XY=tformfwd(WZ,tform2)
WZ2=tforminv(XY,tform2)
%目测检验给定的变换
figure,vistform(tform1,pointgrid([0 0;100 100]))
figure,vistform(tform2,pointgrid([0 0;100 100]))
%% 例5.2 图像的几何变换
clc,clear,close all;
f=checkerboard(50);
sx=0.75;
sy=1.25;
T=[sx 0 0
    0 sy 0
    0 0 1];
t1=maketform('affine',T);
g1=imtransform(f,t1);
theta=pi/6;
T2=[cos(theta) sin(theta) 0
    -sin(theta) cos(theta) 0
    0 0 1];
t2=maketform('affine',T2);
g2=imtransform(f,t2);
T3= [0.4788 0.0135 -0.0009
    0.0135 0.4788 -0.0009
    0.5059 0.5059 1.0000];
tform3=maketform('projective',T3);
g3=imtransform(f,tform3);
figure
subplot(221),imshow(f, []),title('原始图像')
subplot(222),imshow(g1, []),title('仿射缩放变换')
subplot(223),imshow(g2, []),title('仿射旋转变换')
subplot(224),imshow(g3, []),title('投影变换')
%% 例5.3在相同的坐标系统中一起显示输入和输出图像
clc,clear,close all;
f=imread('Fig0508.tif');
theta=3*pi/4;
T=[cos(theta) sin(theta) 0
-sin(theta) cos(theta) 0
0 0 1];
tform=maketform('affine',T);
[g,xdata,ydata]=imtransform(f,tform,'FillValue',255);
figure
subplot(221),imshow(f),axis on
subplot(222),imshow(f),hold on,imshow(g,'XData',xdata,'Ydata',ydata),axis auto,axis on
T=[1 0 0;0 1 0;1000 400 1];
tform=maketform('affine',T);
g=imtransform(f,tform);
[g,xdata,ydata]=imtransform(f,tform,'FillValue',255);
subplot(223),imshow(g),axis on
subplot(224),imshow(f),hold on,imshow(g,'XData',xdata,'Ydata',ydata),axis auto,axis on
%% 例5.4使用函数imtransform2
clc,clear,close all;
f=imread('Fig0513(a).tif');
figure
subplot(231),imshow(f)
tform1=maketform('affine',[1 0 0;0 1 0;300 500 1]);
g1=imtransform2(f,tform1,'FillValue',255);
h1=imtransform(f,tform1,'FillValue',255);
subplot(232),imshow(g1)
subplot(233),imshow(h1)
tform2=maketform('affine',[0.25 0 0;0 0.25 0;0 0 1]);
g2=imtransform2(f,tform2,'FillValue',255);
h2=imtransform(f,tform2,'FillValue',255);
subplot(234),imshow(g2)
subplot(235),imshow(h2)
%% 例5.5针对一些内插方法比较速度和图像质量
clc,clear,close all;
f=imread('Fig0517(a).tif');
timeit(@() reprotate(f,'nearest'))%最近邻内插
timeit(@()  reprotate(f,'bilinear'))%双线性内插
timeit(@() reprotate(f,'bicubic'))%双三次内插
subplot(221),imshow(f)
subplot(222),imshow( reprotate(f,'nearest'))
subplot(223),imshow( reprotate(f,'bilinear'))
subplot(224),imshow( reprotate(f,'bicubic'))
%% 图像配准
f=imread('Fig0518(a).tif');
g=imread('Fig0518(b).tif');
cpselect(f,g)

%% 例5.6使用xisreg观察配准后的图像
clc,clear,close all;
f=imread('Fig0518(a).tif');
fref=imread('Fig0518(b).tif');
s=load('cpselect-results');
cpstruct=s.cpstruct;
tform=cp2tform(cpstruct,'affine');
visreg(fref,f,tform)
%% 例5.7使用函数normxcorr2在图像中定位模板
clc,clear,close all;
f=imread('Fig0521(a).tif');
w=imcrop(f);
g=normxcorr2(w,f);
imshow(abs(g))
gabs=abs(g);
[ypeak,xpeak]=find(gabs==max(gabs(:)));
ypeak=ypeak-(size(w,1)-1)/2;
xpeak=xpeak-(size(w,2)-1)/2;
imshow(f)
hold on
plot(xpeak,ypeak,'wo')
%% 例5.8使用normxcorr2配准由于平移而不同的两幅图像
clc,clear,close all;
f1=imread('Fig0521(a).tif');
f2=imread('Fig0521(b).tif');
w=imcrop(f1);
g1=normxcorr2(w,f1);
g2=normxcorr2(w,f2);
[y1,x1]=find(g1==max(g1(:)));
[y2,x2]=find(g2==max(g2(:)));
delta_x=x1-x2;
delta_y=y1-y2;
tform=maketform('affine',[1 0 0;0 1 0;delta_x delta_y 1]);
visreg(f1,f2,tform)











  




