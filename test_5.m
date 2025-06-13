%% ��5.1 �����Զ����tform�ṹ���������任��
% �任����1������3ˮƽ�ķŴ�����2��ֱ�ķŴ�
clc,clear,close all;
forward_fcn=@(wz,tdata)[3*wz(:,1),2*wz(:,2)];%������
inverse_fcn=@(xy,tdata)[xy(:,1)/3,xy(:,2)/2];%������
tform1=maketform('custom',2,2,forward_fcn,inverse_fcn,[])%�����ռ�任�ṹ
WZ=[1 1;3 2];
XY=tformfwd(WZ,tform1)
WZ2=tforminv(XY,tform1)
%�任����2����ֱ���������ƶ�ˮƽ���꣬�����ִ�ֱ���겻��
forward_fcn=@(wz,tdata)[wz(:,1)+0.4*wz(:,2),wz(:,2)];%������
inverse_fcn=@(xy,tdata)[xy(:,1)-0.4*xy(:,2),xy(:,2)];%������
tform2=maketform('custom',2,2,forward_fcn,inverse_fcn,[])%�����ռ�任�ṹ
XY=tformfwd(WZ,tform2)
WZ2=tforminv(XY,tform2)
%Ŀ���������ı任
figure,vistform(tform1,pointgrid([0 0;100 100]))
figure,vistform(tform2,pointgrid([0 0;100 100]))
%% ��5.2 ͼ��ļ��α任
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
subplot(221),imshow(f, []),title('ԭʼͼ��')
subplot(222),imshow(g1, []),title('�������ű任')
subplot(223),imshow(g2, []),title('������ת�任')
subplot(224),imshow(g3, []),title('ͶӰ�任')
%% ��5.3����ͬ������ϵͳ��һ����ʾ��������ͼ��
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
%% ��5.4ʹ�ú���imtransform2
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
%% ��5.5���һЩ�ڲ巽���Ƚ��ٶȺ�ͼ������
clc,clear,close all;
f=imread('Fig0517(a).tif');
timeit(@() reprotate(f,'nearest'))%������ڲ�
timeit(@()  reprotate(f,'bilinear'))%˫�����ڲ�
timeit(@() reprotate(f,'bicubic'))%˫�����ڲ�
subplot(221),imshow(f)
subplot(222),imshow( reprotate(f,'nearest'))
subplot(223),imshow( reprotate(f,'bilinear'))
subplot(224),imshow( reprotate(f,'bicubic'))
%% ͼ����׼
f=imread('Fig0518(a).tif');
g=imread('Fig0518(b).tif');
cpselect(f,g)

%% ��5.6ʹ��xisreg�۲���׼���ͼ��
clc,clear,close all;
f=imread('Fig0518(a).tif');
fref=imread('Fig0518(b).tif');
s=load('cpselect-results');
cpstruct=s.cpstruct;
tform=cp2tform(cpstruct,'affine');
visreg(fref,f,tform)
%% ��5.7ʹ�ú���normxcorr2��ͼ���ж�λģ��
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
%% ��5.8ʹ��normxcorr2��׼����ƽ�ƶ���ͬ������ͼ��
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











  




