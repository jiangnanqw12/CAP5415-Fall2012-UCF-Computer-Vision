clear all;
close all;

%�ú������ݸ���������ɢԭ�����������Եĳ߶ȿռ䣬����harris���нǵ���

%% ���벢��ʾ�ο��ʹ���׼ͼ��

[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'ѡ��ο�ͼ��ʹ���׼ͼ��',...
                          'F:\class_file\ͼ����׼\ͼ����׼');%ѡ����ͼ����������
image_1=imread(strcat(pathname,filename));
[filename,pathname]=uigetfile({'*.*','All Files(*.*)'},'ѡ��ο�ͼ��ʹ���׼ͼ��',...
                          'F:\class_file\ͼ����׼\ͼ����׼');%ѡ����ͼ����������
image_2=imread(strcat(pathname,filename));

figure;
subplot(2,1,1);
imshow(image_1);
title('�ο�ͼ��');
subplot(2,1,2);
imshow(image_2);
title('����׼ͼ��');
%button=questdlg('�Ƿ���ʾ�м���ͼ������ݣ�','��ʾѡ��','YES','NO','YES');
button='NO';

t1=clock;
%% ��ʼ�����趨
sigma_1=1.6;%��һ��ĳ߶�
sigma_2=1;
ratio=2^(1/3);%�߶ȱ�
nbin=500;%����Աȶ�����ʱ�򹹽����ݶ�ֱ��ͼ��Bin����
perc=0.7;%����Աȶ�����ʱ��İٷ�λ,���ֵԽ��ƽ��Խ��
Mmax=8;%�߶ȿռ�Ĳ���
which_diff=2;%ѡ�������ɢϵ���ķ���
is_auto='YES';%�Ƿ��Զ�����Աȶ���ֵk
first_layer=1;%��ֵ���⿪ʼ����

d=0.04;%HARRIS�������ⳣ��Ĭ����0.04
d_SH_1=0.1;%�ο�ͼ����ֵ�����scharr�˲�ʱ��ȡֵ�ϴ�500�������sobel�˲�ȡֵ��С
d_SH_2=0.1;%����׼ͼ����ֵ

change_form='���Ʊ任';%���������Ʊ任������任��
sift_or_log_polar='����������������';%�����ǡ����������������ӡ��͡�SIFT�����ӡ�

%% ת������ͼ���ʽ
[~,~,num1]=size(image_1);
[~,~,num2]=size(image_2);
if(num1==3)
    image_11=rgb2gray(image_1);
else
    image_11=image_1;
end
if(num2==3)
    image_22=rgb2gray(image_2);
else
    image_22=image_2;
end

%ת��Ϊ��������
image_11=im2double(image_11);
image_22=im2double(image_22);                   

%% ͼ�������������֤�㷨��������Ӱ��
%ͼ�������������������Դ���׼ͼ���������
% button=questdlg('ѡ����������','��ʾѡ��','��˹����',....
%     '��������','��������','��˹����');
% if(strcmp(button,'��˹����'))
%     prompt={'�����˹������ֵ:','�����˹��������:'};
%     dlg_title='��˹������������';
%     def={'0','0.01'};%Ĭ�ϵľ�ֵ��0��������0.01
%     numberlines=1;
%     answer=str2double(inputdlg(prompt,dlg_title,numberlines,def));
%     image_22=imnoise(image_22,'gaussian',answer(1),answer(2));
% elseif(strcmp(button,'��������'))
%     prompt={'�������Pa:'};
%     dlg_title='����������������';
%     def={'0.1'};%%Ĭ����0.05
%     numberlines=1;
%     answer=str2double(inputdlg(prompt,dlg_title,numberlines,def));
%     image_22=imnoise(image_22,'salt & pepper',answer);
% elseif(strcmp(button,'��������'))
%     prompt={'���뷽��:'};
%     dlg_title='����������������';
%     def={'0.1'};%Ĭ����0.04
%     numberlines=1;
%     answer=str2double(inputdlg(prompt,dlg_title,numberlines,def));
%     image_22=imnoise(image_22,'speckle',answer);
% else 
%     image_22=image_22;
% end
% figure;
% subplot(1,2,1);
% imshow(image_2);
% title('����׼ͼ��');
% subplot(1,2,2);
% imshow(image_22);
% title(['����',button,'�Ĵ���׼ͼ��']);


%% ���������Գ߶ȿռ䣬������������˳߶ȿռ䣬
tic;
[nonelinear_space_1]=create_nonlinear_scale_space(image_11,sigma_1,sigma_2,ratio,...
                                 Mmax,nbin,perc,which_diff,is_auto);
[nonelinear_space_2]=create_nonlinear_scale_space(image_22,sigma_1,sigma_2,ratio,...
                                 Mmax,nbin,perc,which_diff,is_auto);
disp(['����������Գ߶ȿռ仨��ʱ�䣺',num2str(toc),'��']);

%% ��������ĸ������Գ߶ȿռ�����Harris�߶ȿռ䣬�߶ȿռ��ÿ��ͼ���ʾharris����
tic;
[harris_function_1,gradient_1,angle_1,]=...
    harris_scale(nonelinear_space_1,d,sigma_1,ratio);  
[harris_function_2,gradient_2,angle_2]=...
    harris_scale(nonelinear_space_2,d,sigma_1,ratio);                                                                                          
disp(['����HARRIS�����߶ȿռ仨��ʱ�䣺',num2str(toc),'��']);

%% ��ʾ�������ɵĽ������
if(strcmp(button,'YES'))
    display_product_image(nonelinear_space_1,gradient_1,angle_1,harris_function_1,'�ο�');
    display_product_image(nonelinear_space_2,gradient_2,angle_2,harris_function_2,'����׼');                                                                                
end                                          

%% ��SAR-HARRIS�����в��Ҽ�ֵ��
tic;
[position_1]=find_scale_extreme(harris_function_1,d_SH_1,sigma_1,ratio,...
             gradient_1,angle_1,first_layer);
[position_2]=find_scale_extreme(harris_function_2,d_SH_2,sigma_1,ratio,...
             gradient_2,angle_2,first_layer);
disp(['�߶ȿռ���Ҽ�ֵ�㻨��ʱ�䣺',num2str(toc),'��']);

%% ��ʾ��⵽�Ľǵ��λ���ڲο�ͼ��ʹ���׼ͼ����
% showpoint_detected(image_1,image_2,position_1,position_2);

%% ����ο�ͼ��ʹ���׼ͼ���������
tic;
[descriptors_1,locs_1]=calc_descriptors(gradient_1,angle_1,...
                                        position_1,sift_or_log_polar);                                     
[descriptors_2,locs_2]=calc_descriptors(gradient_2,angle_2,...
                                        position_2,sift_or_log_polar);   
disp(['��������������ʱ�䣺',num2str(toc),'��']);
                                              
%% ��ʼƥ��
tic;
[solution,rmse,cor1,cor2]=match(image_2, image_1,...
                                descriptors_2,locs_2,...
                                descriptors_1,locs_1,change_form);
    
tform=maketform('projective',solution');
% tform=maketform('affine',solution');
[M,N,P]=size(image_1);
ff=imtransform(image_2,tform, 'XData',[1 N], 'YData',[1 M]);
f=figure;
subplot(1,2,1);
imshow(image_1);
title('�ο�ͼ��');
subplot(1,2,2);
imshow(ff);
title('��׼���ͼ��');
disp(['������ƥ�仨��ʱ�䣺',num2str(toc),'��']);
%����
str1=['.\save_image\','�ο�����׼���ͼ��','.jpg'];
saveas(f,str1,'jpg');
str=['.\save_image\','��׼��ͼ��.jpg'];
imwrite(ff,str,'jpg');
str=['.\save_image\','�ο�ͼ��.jpg'];
imwrite(image_1,str,'jpg');
str=['.\save_image\','����׼ͼ��.jpg'];
imwrite(image_2,str,'jpg');

%% ͼ���ں�
image_fusion(image_1,image_2,solution)
t2=clock;
disp(['������ʱ���ǣ�',num2str(etime(t2,t1)),'��']);                                              

%% ��ʾ��ֲ�
button=disp_points_distribute_1(locs_1,locs_2,cor2,cor1,Mmax);
% %����
% str1=['.\save_image\','����ֲ�','.jpg'];
% saveas(button,str1,'jpg');
% 
% showpoint_detected(image_1,image_2,cor2,cor1);                                             
                                              
                                              
                                              
                                              
                                              
                                              
                                              
                                              
