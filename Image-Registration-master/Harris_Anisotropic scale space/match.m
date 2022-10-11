function [solution,rmse,cor1,cor2]= match(im1, im2,des1,loc1,...
       des2,loc2,change_form)
%% ����im1�Ǵ���׼ͼ��im2�ǲο�ͼ��
%% ��ʼƥ��
distRatio=0.9;
des2t = des2';

%���ڲο�ͼ���е�ÿһ����Ѱ�Һʹ�ƥ��ͼ���е����Ƶ�
for i = 1 : size(des1,1)
  dotprods = des1(i,:) * des2t;        
  [vals,indx] = sort(acos(dotprods));  
  if (vals(1) < distRatio * vals(2))
     match(i) = indx(1);%%match������des2t�еĶ�Ӧ�������������
  else
      match(i) = 0;
  end
end

%����ο�ͼ��ʹ���׼ͼ�������������
fprintf('�ο�ͼ��������������Ŀ%d.\n����׼ͼ��������������Ŀ��%d.\n', size(des2,1),size(des1,1));
num = sum(match > 0);%%ƥ��ĸ���
fprintf('��ʼ�����Found %d matches.\n', num);
[~,point1,point2]=find(match);
%���桾x,y,�߶ȣ�layer���Ƕȡ�
cor1=loc1(point1,[1 2 3 4 5]);
cor2=loc2(point2,[1 2 3 4 5]);
cor1=[cor1 point2'];cor2=[cor2 point2'];%point2�������ʼ����������

%% �Ƴ��ظ����
uni1=[cor1(:,[1,2]),cor2(:,[1,2])];
[~,i,~]=unique(uni1,'rows','first');
cor1=cor1(sort(i)',:);cor2=cor2(sort(i)',:);
fprintf('ɾ���ظ���Ժ�Found %d matches.\n', size(cor1,1));

%% ��ʼ�Ƴ������Ժ�ʹ��ransac�㷨
[solution,rmse,cor1,cor2]=ransac(cor1,cor2,change_form,1);
fprintf('Ransacɾ�������Ժ� %d matches.\n', size(cor1,1));

%% ���������ȷ��Ե��                                       
fprintf('���Found %d matches.\n', size(cor1,1));
[hand_1,hand_2]=showpoints(im2,im1,cor2,cor1);
str1=['.\save_image\','�ο�ͼ����ȷ������.jpg'];
saveas(hand_1,str1,'jpg');
str1=['.\save_image\','����׼ͼ����ȷ��Ե�.jpg'];
saveas(hand_2,str1,'jpg');

fhand=appendimages(im2,im1,cor2,cor1);
str1=['.\save_image\','����ƥ����.jpg'];
saveas(fhand,str1,'jpg');

cor1=cor1(:,1:5);
cor2=cor2(:,1:5);



