function showpoint_detected(im1,im2,cor1,cor2)
%% ����im1�ǲο�ͼ��im2�Ǵ���׼ͼ��,
%�ú�������������ʾ�ڲο�ͼ��ʹ���׼ͼ���ϼ�⵽�Ľǵ�
%��Ϊһ������������ж�����������ɾ���ظ�������������
uni1=cor1(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);
cor1_x=cor1(:,1);cor1_y=cor1(:,2);
f=figure;colormap('gray');imagesc(im1);
title(['�ο�ͼ��',num2str(size(cor1_x,1)),'������']);hold on;
scatter(cor1_x,cor1_y,'r');hold on;%scatter���������ɢ��ͼ
str1=['.\save_image\','�ο�ͼ����������','.jpg'];
saveas(f,str1,'jpg');
fprintf('�ο�ͼ���⵽�����������%d\n', size(cor1,1));

uni1=cor2(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor2=cor2(sort(i)',:);
cor2_x=cor2(:,1);cor2_y=cor2(:,2);
f=figure;colormap('gray');imagesc(im2);
title(['����׼ͼ��',num2str(size(cor2_x,1)),'������']);hold on;
scatter(cor2_x,cor2_y,'r');hold on;
str1=['.\save_image\','����׼ͼ����������','.jpg'];
saveas(f,str1,'jpg');
fprintf('����׼ͼ���⵽�����������%d\n', size(cor2,1));


uni1=cor1(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);
cor1_x=cor1(:,1);cor1_y=cor1(:,2);
figure;
imshow(im1,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,size(im1,2) size(im1,1)]);
axis normal;
hold on;
scatter(cor1_x,cor1_y,'r','*');hold on;%scatter���������ɢ��ͼ
for i=1:size(cor1,1)
text(cor1_x(i),cor1_y(i),num2str(i),'color','b');
end

uni1=cor2(:,[1,2,3,4]);
[~,i,~]=unique(uni1,'rows','first');
cor2=cor2(sort(i)',:);
cor2_x=cor2(:,1);cor2_y=cor2(:,2);
figure;
imshow(im2,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,size(im2,2) size(im2,1)]);
axis normal;
hold on;
scatter(cor2_x,cor2_y,'r','*');hold on;%scatter���������ɢ��ͼ
for i=1:size(cor1,1)
text(cor2_x(i),cor2_y(i),num2str(i),'color','b');
end


end



