#ifndef _match_h_
#define _match_h_

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

#include<vector>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;

const double dis_ratio = 0.8;//����ںʹν��ھ������ֵ
const float ransac_error = 1.5;//ransac�㷨�����ֵ

enum DIS_CRIT{ Euclidean=0,COS};//�������׼��

/*�ú�����������׼���ͼ������ں���Ƕ*/
void image_fusion(const Mat &image_1, const Mat &image_2, const Mat T, Mat &fusion_image, Mat &mosaic_image, Mat &matched_image);

/*�ú������������ӵ�����ںʹν���ƥ��*/
void match_des(const Mat &des_1, const Mat &des_2, vector<vector<DMatch>> &dmatchs, DIS_CRIT dis_crite);

/*�ú���ɾ������ƥ���ԣ���������׼*/
Mat match(const Mat &image_1, const Mat &image_2, const vector<vector<DMatch>> &dmatchs, vector<KeyPoint> keys_1,
	vector<KeyPoint> keys_2, string model, vector<DMatch> &right_matchs, Mat &matched_line);

#endif
