%% 图像读入及显示
f=imread('C:\Users\Asus\Desktop\数字图像处理\image1.jpg');
imshow(f)
%% 图像写入，新窗口显示(JPEG 压缩文件的质量，指定为范围 [0, 100] 内的标量，其中 0 表示较低质量和较高的压缩率，100 表示较高质量和较低的压缩率。)
imwrite(f,'image1.jpg','quality',10);
f1=imread('image1.jpg');
figure,imshow(f1);
%% 图像类型转换（二值图像是取值只有0和1的逻辑数组）
f2=imread('Fig0101.tif');
imshow(f2)
B=logical(f2);
islogical(B)
figure,imshow(B)
%%B=im2uint8(B);
%% I = mat2gray(A,[amin amax]) 将矩阵 A 转换为灰度图像 I，该图像包含 0（黑色）到 1（白色）范围内的值。amin 和 amax 是 A 中对应于 I 中 0 和 1 的值。小于 amin 的值裁剪到 0，大于 amax 的值裁剪到 1。
f3=mat2gray(f2);
imshow(f3)
%% 向量和矩阵及索引
V=[1:2:9] 
V(2) %数组索引
W=V.'
W(3:end)
V([1 4 5])
V(1:2:end)
A=[1 2 3;4 5 6 ;7 8 9 ]
A1=A(:) %将矩阵转换为列向量
sum(A1) %sum函数，参量为向量则元素求和，参量为矩阵则计算各列向量元素求和
D=logical([1 0 0;0 1 0;0 0 1])
A(D) %逻辑索引（D是与A大小相同的逻辑数组）             线性索性：用单个下标来编制高维矩阵或数组的索引
%% 函数句柄
f_1=@sin; %命名（简单）的函数句柄
f_1(pi/4)
g=@(x) x.^2; %匿名的函数句柄
r=@(x,y) sqrt(x.^2+y.^2);
r(1,2)
%% 单元数组，单一变量名下组合混合的一组对象  （包含参数副本而不是指针，数字寻址）
char_array={'ning','yi'};
C={f,f2,char_array}
C{3}%检索元素
C(3)%检索元素属性
%% 结构（字段寻址）
s=image_stats(f2)
s.dm %结构体的检索
%% 优化编码（1.预分配空间    2.向量化即去循环）
sinfun1(5)
tic; sinfun1(100); toc  %测量函数执行时间
M=100;
f_2=@() sinfun1(M);
timeit(f_2) %（为了执行稳健的测量，timeit 多次调用指定的函数，并返回测量结果的中位数。如果该函数运行速度很快，timeit 可能会多次调用该函数。）

M=500:500:20000;
for k=1:numel(M)
    f1=@()sinfun1(M(k));
    f2=@()sinfun2(M(k));%预分配
    f3=@()sinfun3(M(k));%向量化完全消除循环
    t1(k)=timeit(f1);
    t2(k)=timeit(f2);
    t3(k)=timeit(f3);
end
figure,plot(M,[t1;t2;t3])
xlabel('M'),ylabel('时间(秒)'),title('运行sinfun1,sinfun2,sinfun3所需时间');
legend('sinfun1','sinfun2','sinfun3');
%%
q1=timeit(@()twodsin1(1,1/(4*pi),1/(4*pi),512,512));%函数含预分配
q2=timeit(@()twodsin2(1,1/(4*pi),1/(4*pi),512,512));%函数含向量化 %[X,Y] = meshgrid(x,y) 基于向量 x 和 y 中包含的坐标返回二维网格坐标。X 是一个矩阵，每一行是 x 的一个副本；Y 也是一个矩阵，每一列是 y 的一个副本。坐标 X 和 Y 表示的网格有 length(y) 个行和 length(x) 个列。
fprintf('twodsin1用时%d，twodsin2用时%d',q1,q2);
f=twodsin1(1,1/(4*pi),1/(4*pi),512,512);
imshow(f,[])%imshow(I,[]) 显示灰度图像 I，根据 I 中的像素值范围对显示进行转换。imshow 使用 [min(I(:)) max(I(:))] 作为显示范围。imshow 将 I 中的最小值显示为黑色，将最大值显示为白色。



 











