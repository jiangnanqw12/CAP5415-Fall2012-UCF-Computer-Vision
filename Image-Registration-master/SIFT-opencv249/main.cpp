#include"sift.h"
#include"display.h"
#include"match.h"

#include<opencv2\highgui\highgui.hpp>
#include<opencv2\calib3d\calib3d.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#include<fstream>
#include<stdlib.h>
#include<direct.h>


#pragma comment( lib,"IlmImf.lib" ) 
#pragma comment( lib,"libjasper.lib" )   
#pragma comment( lib,"libjpeg.lib" )   
#pragma comment( lib,"libpng.lib" )   
#pragma comment( lib,"libtiff.lib" )   
#pragma comment( lib, "zlib.lib") 
#pragma comment( lib, "opencv_ts249.lib") 

#pragma comment( lib, "vfw32.lib" )   
#pragma comment( lib, "comctl32.lib" ) 
#pragma comment( lib, "libcmt.lib" ) 


#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_features2d249.lib")
#pragma comment(lib,"opencv_flann249.lib")
#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")

int main(int argc,char *argv[])
{
	// string change_model;
	//if (argc <3 || argc>4){//�����������������3����4
	//	cout << "********�����������,���������������ĸ���ȷ������********" << endl;
	//	cout << "�밴������˳��Ҫ�����������" << endl;
	//	cout << "1. ��ִ���ļ�" << endl;
	//	cout << "2. �ο�ͼ��" << endl;
	//	cout << "3. ����׼ͼ��" << endl;
	//	cout << "4. �任����" << endl;
	//	cout << "�������ӣ�" << endl;
	//	cout << "ʵ��1�� sift_2.exe school_3.jpg school_4.jpg similarity" << endl;
	//	cout << "ʵ��2�� sift_2.exe school_3.jpg school_4.jpg affine" << endl;
	//	cout << "ʵ��3�� sift_2.exe school_3.jpg school_4.jpg perspective" << endl;
	//	cout << "ʵ��4�� sift_2.exe school_3.jpg school_4.jpg" << endl;
	//	cout << "************************************************************" << endl;
	//	return -1;
	//}
	//else if (argc == 3)//�����3��������Ĭ��ѡ��͸�ӱ任ģ��
	//	change_model = string("perspective");
	//else if (argc == 4){
	//	change_model = string(argv[3]);//��������"similarity","perspective","affine"
	//	if (!(change_model == string("affine") || change_model == string("similarity") ||
	//		change_model == string("perspective")))
	//	{
	//		cout << "********�任�����������********" << endl;
	//		return -1;
	//	}
	//}

	////��������
	//Mat image_1, image_2;
	//image_1 = imread(argv[1], -1);
	//image_2 = imread(argv[2], -1);
	//if (!image_1.data || !image_2.data){
	//	cout << "ͼ�����ݼ���ʧ�ܣ�" << endl;
	//	return -1;
	//}

	/*Mat image_1 = imread("E:\\class_file\\graduate_data\\ͼ����׼\\sift\\siftOpencv\\sift_static\\Debug\\ucsb1.jpg", -1);
	Mat image_2 = imread("E:\\class_file\\graduate_data\\ͼ����׼\\sift\\siftOpencv\\sift_static\\Debug\\ucsb2.jpg", -1);*/

	//system("cd ..\\..\\");//������һ��

	Mat image_1 = imread("..\\..\\set\\ucsb1.jpg", -1);
	Mat image_2 = imread("..\\..\\set\\ucsb2.jpg", -1);
	string change_model = "perspective";

	//�����ļ��б���ͼ��
	char* newfile = ".\\image_save";
	_mkdir(newfile);

	//�㷨������ʱ�俪ʼ��ʱ
	double total_count_beg = (double)getTickCount();

	//�����
	MySift sift_1(0, 3, 0.04, 10, 1.6, true);

	//�ο�ͼ���������������
	vector<vector<Mat>> gauss_pyr_1, dog_pyr_1;
	vector<KeyPoint> keypoints_1;
	Mat descriptors_1;
	double detect_1 = (double)getTickCount();
	sift_1.detect(image_1, gauss_pyr_1, dog_pyr_1, keypoints_1);
	double detect_time_1 = ((double)getTickCount() - detect_1) / getTickFrequency();
	cout << "�ο�ͼ����������ʱ���ǣ� " << detect_time_1 << "s" << endl;
	cout << "�ο�ͼ��������������ǣ� " << keypoints_1.size() << endl;

	double comput_1 = (double)getTickCount();
	sift_1.comput_des(gauss_pyr_1, keypoints_1, descriptors_1);
	double comput_time_1 = ((double)getTickCount() - comput_1) / getTickFrequency();
	cout << "�ο�ͼ������������ʱ���ǣ� " << comput_time_1 << "s" << endl;


	//����׼ͼ���������������
	vector<vector<Mat>> gauss_pyr_2, dog_pyr_2;
	vector<KeyPoint> keypoints_2;
	Mat descriptors_2;
	double detect_2 = (double)getTickCount();
	sift_1.detect(image_2, gauss_pyr_2, dog_pyr_2, keypoints_2);
	double detect_time_2 = ((double)getTickCount() - detect_2) / getTickFrequency();
	cout << "����׼ͼ����������ʱ���ǣ� " << detect_time_2 << "s" << endl;
	cout << "����׼ͼ��������������ǣ� " << keypoints_2.size() << endl;

	double comput_2 = (double)getTickCount();
	sift_1.comput_des(gauss_pyr_2, keypoints_2, descriptors_2);
	double comput_time_2 = ((double)getTickCount() - comput_2) / getTickFrequency();
	cout << "����׼����������ʱ���ǣ� " << comput_time_2 << "s" << endl;

	//�������ν��ھ����ƥ��
	double match_time = (double)getTickCount();
	Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher;
	//Ptr<DescriptorMatcher> matcher = new BFMatcher(NORM_L2);
	std::vector<vector<DMatch>> dmatchs;
	matcher->knnMatch(descriptors_1, descriptors_2, dmatchs, 2);
	//match_des(descriptors_1, descriptors_2, dmatchs, COS);

	Mat matched_lines;
	vector<DMatch> right_matchs;
	Mat homography = match(image_1, image_2, dmatchs, keypoints_1, keypoints_2, change_model,
		right_matchs,matched_lines);
	double match_time_2 = ((double)getTickCount() - match_time) / getTickFrequency();
	cout << "������ƥ�仨��ʱ���ǣ� " << match_time_2 << "s" << endl;
	cout << change_model << "�任�����ǣ�" << endl; 
	cout << homography << endl;

	//����ȷƥ�������д���ļ���
	ofstream ofile;
	ofile.open(".\\position.txt");
	for (size_t i = 0; i < right_matchs.size(); ++i)
	{
		ofile << keypoints_1[right_matchs[i].queryIdx].pt << "   "
			<< keypoints_2[right_matchs[i].trainIdx].pt << endl;
	}

	//ͼ���ں�
	double fusion_beg = (double)getTickCount();
	Mat fusion_image, mosaic_image, regist_image;
	image_fusion(image_1, image_2, homography, fusion_image, mosaic_image, regist_image);
	imwrite(".\\image_save\\�ںϺ��ͼ��.jpg", fusion_image);
	imwrite(".\\image_save\\�ںϺ����Ƕͼ��.jpg", mosaic_image);
	imwrite(".\\image_save\\��׼��Ĵ���׼ͼ��.jpg", regist_image);
	double fusion_time = ((double)getTickCount() - fusion_beg) / getTickFrequency();
	cout << "ͼ���ںϻ���ʱ���ǣ� " << fusion_time << "s" << endl;

	double total_time = ((double)getTickCount() - total_count_beg) / getTickFrequency();
	cout << "�ܻ���ʱ���ǣ� " << total_time << "s" << endl;

	//��ʾƥ����
	namedWindow("�ںϺ��ͼ��", WINDOW_AUTOSIZE);
	imshow("�ںϺ��ͼ��", fusion_image);
	namedWindow("�ں���Ƕͼ��", WINDOW_AUTOSIZE);
	imshow("�ں���Ƕͼ��", mosaic_image);
	stringstream s_2;
	string numstring_2, windowName;
	s_2 << right_matchs.size();
	s_2 >> numstring_2;
	windowName = string("��ȷ��ƥ������ͼ: ") + numstring_2;
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, matched_lines);

	//���������ƴ�ӺõĽ�����ͼ��
	//int nOctaveLayers = sift_1.get_nOctave_layers();
	//write_mosaic_pyramid(gauss_pyr_1, dog_pyr_1, gauss_pyr_2, dog_pyr_2, nOctaveLayers);

	waitKey(0);
	return 0;
}