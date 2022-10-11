function  [fhand]=appendimages(image1, image2,correspond1,correspond2)

% if(size(image1,3)==3)
%     image1=rgb2gray(image1);
% end
% if(size(image2,3)==3)
%     image2=rgb2gray(image2);
% end
% Select the image with the fewest rows and fill in enough empty rows
%   to make it the same height as the other image.
rows1 = size(image1,1);
rows2 = size(image2,1);

col1=size(image1,2);
col2=size(image2,2);
% if (rows1 < rows2)
%      image1(rows2,1) = 0;
% else
%      image2(rows1,1) = 0;
% end
if (rows1 < rows2)
     image1(rows1+1:rows2,1:col1,:) = 0;
elseif(rows1 >rows2)
     image2(rows2+1:rows1,1:col2,:) = 0;
end
% Now append both images side-by-side.
im3 = [image1 image2]; 

%��һ�ַ���ɾ���ױ�
% fhand=figure('Position', [100 100 size(im3,2) size(im3,1)]);
% imshow(im3);

%�ڶ��з���ɾ���ױ�
fhand=figure;
imshow(im3,'border','tight','initialmagnification','fit');
%title(['����ǲο�ͼ��---�����Ŀ',num2str(size(correspond1,1)),'---�Ҳ��Ǵ���׼ͼ��']);
set (gcf,'Position',[0,0,size(im3,2) size(im3,1)]);
axis normal;

%�����ַ���ɾ���ױ�
% set(0,'CurrentFigure',fhand);
% set(gcf,'PaperPositionMode','auto');
% set(gca,'position',[0,0,1,1]);
% set(gcf,'position',[1,1,size(im3,2) size(im3,1)]);

hold on;
cols1 = size(image1,2);
for i = 1: size(correspond1,1)
    num=round(1+(3-1)*rand(1,1));
    num=1;%�̶���ɫ
    if(num==1)%��ɫ
        line([correspond1(i,1) correspond2(i,1)+cols1], ...
             [correspond1(i,2) correspond2(i,2)], 'Color', 'r','LineWidth',1);
    elseif(num==2)%��ɫ
        line([correspond1(i,1) correspond2(i,1)+cols1], ...
             [correspond1(i,2) correspond2(i,2)], 'Color', 'g','LineWidth',1); 
    elseif(num==3)%��ɫ
        line([correspond1(i,1) correspond2(i,1)+cols1], ...
             [correspond1(i,2) correspond2(i,2)], 'Color', 'b','LineWidth',1); 
    end
end

hold off;

end






