function display_product_image(gaussian_image,...
                                gaussian_gradient_second,...
                                gaussian_angle_second,...
                                gauss_harris_fun,str)
%%�ú�����Ҫ��ɳ����м������ͼ�����ʾ�ʹ洢����
%gaussian_image�ǳ߶ȿռ�ÿ��ĵĸ�˹ͼ��
%gaussian_harris_fun�ǳ߶ȿռ�ÿ���harris����
%str��ʾ�ǲο�ͼ���Ǵ���׼ͼ��
[~,~,Mmax]=size(gaussian_image);
temp=Mmax/4;
if((floor(temp)-temp)~=0)
    temp=temp+1;
end
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
    imshow(mat2gray(gaussian_image(:,:,i)));
    title([str,'��������',num2str(i),'��']);
    str1=['.\save_image\',str,'�������Կռ�',num2str(i),'��','.png'];
    imwrite(mat2gray(gaussian_image(:,:,i)),str1,'png');
end
str1=['.\save_image\',str,'�������Կռ�','.jpg'];
saveas(f,str1,'jpg');


% ��ʾͼ���Harris����ͼ��
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
%     min_temp=min(min(gauss_harris_fun(:,:,i)));
%     max_temp=max(max(gauss_harris_fun(:,:,i)));
%     imshow((gauss_harris_fun(:,:,i)-min_temp)/(max_temp-min_temp));
    imshow(mat2gray(gauss_harris_fun(:,:,i)));
    title([str,'HARRIS����',num2str(i),'��']);
end
str1=['.\save_image\',str,'HARRIS����','.jpg'];
saveas(f,str1,'jpg');

%��ʾͼ��Ķ��׸�˹����ݶ�
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
    imshow(mat2gray(gaussian_gradient_second(:,:,i)));
    title([str,'�����ݶ�',num2str(i),'��']);
end
str1=['.\save_image\',str,'���ײ���ݶ�','.jpg'];
saveas(f,str1,'jpg');
    

%��ʾ��˹ͼ����ײ�ֽǶ�
f=figure;
for i=1:1:Mmax
    subplot(temp,4,i);
    imshow(mat2gray(gaussian_angle_second(:,:,i)));
    title([str,'���׽Ƕ�',num2str(i),'��']);
end
str1=['.\save_image\',str,'���ײ�ֽǶ�','.jpg'];
saveas(f,str1,'jpg');

end

                            

