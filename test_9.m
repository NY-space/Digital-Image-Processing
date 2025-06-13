%% ��9.1���͵�Ӧ��
clc,clear,close all;
A=imread('Fig0906(a).tif');
B=[0 1 0;1 1 1;0 1 0];%�ṹԪ
D=imdilate(A,B);%����
figure
subplot(121),imshow(A)
subplot(122),imshow(D)
%% ��9.2��strel�ֽ�ṹԪ
clc,clear,close all;
se=strel('diamond',5)
decomp=getsequence(se);%��ȡ�����ֽ��еĵ����ĽṹԪ
whos%�г��������еı�������С������
decomp(1)%�����ֽ���4���ṹԪ
decomp(2)
decomp(3)
decomp(4)
%% ��9.3��ʴ��˵��
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
%% ��9.4ʹ��imopen��imclose�����������ղ�����
clc,clear,close all;
f=imread('Fig0910(a).tif');
se=strel('square',40);
fo=imopen(f,se);
fc=imclose(f,se);
foc=imclose(fo,se);
figure
subplot(221),imshow(f),title('ԭʼͼ��');
subplot(222),imshow(fo),title('������');
subplot(223),imshow(fc),title('�ղ���');
subplot(224),imshow(foc),title('�ȿ����');
%��������
f=imread('Fig0911(a).tif');
se=strel('square',6);
fo=imopen(f,se);
foc=imclose(fo,se);
figure
subplot(131),imshow(f),title('ԭʼͼ��');
subplot(132),imshow(fo),title('������');
subplot(133),imshow(foc),title('�ȿ����');
%% ��9.5ʹ�ú���bwhitmiss�����л�����б任��
clc,clear,close all;
f=imread('Fig0913(a).tif');
B1=strel([0 0 0 ;0 1 1;0 1 0]);
B2=strel([1 1 1;1 0 0;1 0 0]);
g=bwhitmiss(f,B1,B2);
%�������
interval=[-1 -1 -1;-1 1 1;-1 1 0];
g=bwhitmiss(f,B1,B2);
figure
subplot(121),imshow(f);
subplot(122),imshow(g);
%% ��9.6�ö�ֵͼ�񼰻��ڲ��ұ�ļ�����Conway��Game of Life������Ϸ
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
%% ��9.7�������ʾ��ͨ����������
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
%% ��9.8ͨ���ؽ����п�����
clc,clear,close all;
f=imread('Fig0922(a).tif');
fe=imerode(f,ones(51,1));%�����߸�ʴ
fo=imopen(f,ones(51,1));%�����߽��п�����
fobr=imreconstruct(fe,f);%������ͨ���ؽ����еĿ�����
figure
subplot(221),imshow(f)
subplot(222),imshow(fe)
subplot(223),imshow(fo)
subplot(224),imshow(fobr)
%% ��9.9�ÿ������ͱղ�������̬ѧƽ��
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
subplot(326),imshow(fasf);%����˳���˲�
%% ��9.10�Ǿ��ȱ����Ĳ���
clc,clear,close all;
f=imread('Fig0926(a).tif');
se=strel('disk',10);
fo=imopen(f,se);%�����Ǿ�����������
f1=f-fo;
f2=imtophat(f,se);
figure
subplot(141),imshow(f);
subplot(142),imshow(fo);
subplot(143),imshow(f1);
subplot(144),imshow(f2);
%% ��9.11���Ȳⶨ
clc,clear,close all;
f=imread('Fig0925(a).tif');
umpixels=zeros(1,36);
for k=0:35
    se=strel('disk',k);
    fo=imopen(f,se);
    sumpixels(k+1)=sum(fo(:));
end
figure
subplot(121),plot(0:35,sumpixels),xlabel('k'),ylabel('Surface area')%����ڽṹԪ�뾶�ı�������
subplot(122),plot(-diff(sumpixels)),xlabel('k'),ylabel('Surface area reduction')%����ڽṹԪ�뾶�ı����������
%% ��9.12���ؽ���ȥ���ӵı�������
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
subplot(331),imshow(f),title('ԭʼͼ��');
subplot(332),imshow(f_obr),title('�ؽ�������');
subplot(333),imshow(f_o),title('������');
subplot(334),imshow(f_thr),title('�ؽ��Ķ�ñ����');
subplot(335),imshow(f_th),title('��ñ����');
subplot(336),imshow(g_obr),title('ʹ��ˮƽ����ԣ�4�������ؽ��Ŀ�����');
subplot(337),imshow(g_obrd),title('��ˮƽ�߶ԣ�6�����е����ʹ���');
subplot(338),imshow(f2),title('�����ؽ�����');
















;