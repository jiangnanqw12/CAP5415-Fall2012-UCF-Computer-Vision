#include"Sar_sift.h"
#include<opencv2\features2d\features2d.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#include<vector>


/*********�ú������ݳ߶Ⱥʹ��ڰ뾶����ROEWA�˲�ģ��************/
/*size��ʾ�˰뾶����˺˿����2*size+1
 scale��ʾָ��Ȩ�ز���
 kernel��ʾ���ɵ��˲���
 */
static void roewa_kernel(int size, float scale,Mat &kernel)
{
	kernel.create(2 * size + 1, 2 * size + 1, CV_32FC1);
	for (int i = -size; i <= size; ++i)
	{
		float *ptr_k = kernel.ptr<float>(i + size);
		for (int j = -size; j <= size; ++j)
		{
			ptr_k[j + size] = exp(-1.f*(abs(i) + abs(j)) / scale);
		}
	}
}


/*�ú������ݴ��ڰ뾶����׼�����Բ�θ�˹�˲�ģ�壬����������*/
/*size��ʾԲ�뾶
 scale��ʾ��˹��׼��
 gauss_kernel��ʾ���ɵ�Բ�θ�˹��
 ��ά��˹��������ʽ��1/(2*pi*scale*scale)*exp(-(x*x+y*y)/(2*scale*scale))
 */
static void gauss_circle(int size, float scale, Mat &gauss_kernel)
{
	gauss_kernel.create(2 * size + 1, 2 * size + 1, CV_32FC1);
	float exp_temp = -1.f / (2 * scale*scale);
	float sum = 0;
	for (int i = -size; i <= size; ++i)
	{
		float *ptr = gauss_kernel.ptr<float>(i +size);
		for (int j = -size; j <= size; ++j)
		{
			if ((i*i + j*j) <= size*size)
			{
				float value = exp((i*i + j*j)*exp_temp);
				sum += value;
				ptr[j +size] = value;
			}
			else
				ptr[j + size] = 0.f;
				
		}
	}

	//
	for (int i = -size; i <= size; ++i)
	{
		float *ptr = gauss_kernel.ptr<float>(i +size);
		for (int j = -size; j <=size; ++j)
			ptr[j +size] = ptr[j + size] / sum;
	}
}

/*************�ú�������SAR_SIFT�߶ȿռ�*****************/
/*image��ʾ�����ԭʼͼ��
 sar_harris_fun��ʾ�߶ȿռ��Sar_harris����
 amplit��ʾ�߶ȿռ����ص��ݶȷ���
 orient��ʾ�߶ȿռ����ص��ݶȷ���
 */
void Sar_sift::build_sar_sift_space(const Mat &image, vector<Mat> &sar_harris_fun, vector<Mat> &amplit, vector<Mat> &orient)
{
	//ת������ͼ���ʽ
	Mat gray_image;
	if (image.channels() != 1)
		cvtColor(image, gray_image, CV_RGB2GRAY);
	else
		gray_image = image;

	//��ͼ��ת��Ϊ0-1֮��ĸ�������
	Mat float_image;
	//������ת��Ϊ0-1֮��ĸ������ݺ�ת��Ϊ0-255֮��ĸ������ݣ�Ч����һ����
	//gray_image.convertTo(float_image, CV_32FC1, 1.f / 255.f, 0.f);//ת��Ϊ0-1֮��
	gray_image.convertTo(float_image, CV_32FC1, 1, 0.f);//ת��Ϊ0-255֮��ĸ�����

	//�����ڴ�
	sar_harris_fun.resize(Mmax);
	amplit.resize(Mmax);
	orient.resize(Mmax);

	for (int i = 0; i < Mmax; ++i)
	{
		float scale = (float)sigma*(float)pow(ratio, i);//��õ�ǰ��ĳ߶�
		int radius = cvRound(2 * scale);
		Mat kernel;
		roewa_kernel(radius, scale, kernel);

		//�ĸ��˲�ģ������
		Mat W34 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);
		Mat W12 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);
		Mat W14 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);
		Mat W23 = Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_32FC1);

		kernel(Range(radius + 1, 2 * radius + 1), Range::all()).copyTo(W34(Range(radius + 1, 2 * radius + 1), Range::all()));
		kernel(Range(0, radius), Range::all()).copyTo(W12(Range(0, radius), Range::all()));
		kernel(Range::all(), Range(radius + 1, 2 * radius + 1)).copyTo(W14(Range::all(), Range(radius + 1, 2 * radius + 1)));
		kernel(Range::all(), Range(0, radius)).copyTo(W23(Range::all(), Range(0, radius)));

		//�˲�
		Mat M34, M12, M14, M23;
		double eps = 0.00001;
		filter2D(float_image, M34, CV_32FC1, W34, Point(-1, -1), eps);
		filter2D(float_image, M12, CV_32FC1, W12, Point(-1, -1), eps);
		filter2D(float_image, M14, CV_32FC1, W14,Point(-1, -1), eps);
		filter2D(float_image, M23, CV_32FC1, W23, Point(-1, -1), eps);

		//����ˮƽ�ݶȺ���ֱ�ݶ�
		Mat Gx, Gy;
		log((M14) / (M23), Gx);
		log((M34) / (M12), Gy);

		//�����ݶȷ��Ⱥ��ݶȷ���
		magnitude(Gx, Gy, amplit[i]);
		phase(Gx, Gy, orient[i], true);

		//����sar-Harris����
		//Mat Csh_11 = log(scale)*log(scale)*Gx.mul(Gx);
		//Mat Csh_12 = log(scale)*log(scale)*Gx.mul(Gy);
		//Mat Csh_22 = log(scale)*log(scale)*Gy.mul(Gy);

		Mat Csh_11 = scale*scale*Gx.mul(Gx);
		Mat Csh_12 = scale*scale*Gx.mul(Gy);
		Mat Csh_22 = scale*scale*Gy.mul(Gy);//��ʱ��ֵΪ0.8

		//Mat Csh_11 = Gx.mul(Gx);
		//Mat Csh_12 = Gx.mul(Gy);
		//Mat Csh_22 = Gy.mul(Gy);//��ʱ��ֵΪ0.8/100

		//��˹Ȩ��
		float gauss_sigma = sqrt(2.f)*scale;
		int size = cvRound(3 * gauss_sigma);

		Size kern_size(2 * size + 1, 2 * size + 1);
		GaussianBlur(Csh_11, Csh_11, kern_size, gauss_sigma, gauss_sigma);
		GaussianBlur(Csh_12, Csh_12, kern_size, gauss_sigma, gauss_sigma);
		GaussianBlur(Csh_22, Csh_22, kern_size, gauss_sigma, gauss_sigma);

		/*Mat gauss_kernel;//�Զ���Բ�θ�˹��
		gauss_circle(size, gauss_sigma, gauss_kernel);
		filter2D(Csh_11, Csh_11, CV_32FC1, gauss_kernel);
		filter2D(Csh_12, Csh_12, CV_32FC1, gauss_kernel);
		filter2D(Csh_22, Csh_22, CV_32FC1, gauss_kernel);*/

		Mat Csh_21 = Csh_12;

		//����sar_harris����
		Mat temp_add = Csh_11 + Csh_22;
		sar_harris_fun[i] = Csh_11.mul(Csh_22) - Csh_21.mul(Csh_12) - (float)d*temp_add.mul(temp_add);
	}
}


/******************�ú�������������������*********************/
/*amplit��ʾ���������ڲ���ݶȷ���
 orient��ʾ���������ڲ���ݶȷ���0-360��
 point��ʾ����������
 scale��ʾ����������ڲ�ĳ߶�
 hist��ʾ���ɵ�ֱ��ͼ
 n��ʾ������ֱ��ͼbin����
 �ú�������ֱ��ͼ�����ֵ
 */
static float calc_orient_hist(const Mat &amplit, const Mat &orient, Point2f pt, float scale,float *hist,int n)
{
	int num_row = amplit.rows;
	int num_col = amplit.cols;

	Point point(cvRound(pt.x), cvRound(pt.y));
	int radius = cvRound(SAR_SIFT_FACT_RADIUS_ORI*scale);
	radius = min(radius, min(num_row / 2, num_col / 2));
	float gauss_sig = 2 * scale;//��˹��Ȩ��׼��
	float exp_temp = -1.f / (2 * gauss_sig*gauss_sig);

	int radius_x_left = point.x - radius;
	int radius_x_right = point.x + radius;
	int radius_y_up = point.y - radius;
	int radius_y_down = point.y + radius;

	//��ֹԽ��
	if (radius_x_left < 0)
		radius_x_left = 0;
	if (radius_x_right>num_col - 1)
		radius_x_right = num_col - 1;
	if (radius_y_up < 0)
		radius_y_up = 0;
	if (radius_y_down>num_row - 1)
		radius_y_down = num_row - 1;

	//��ʱ��������Χ������������ڱ��������������
	int center_x = point.x - radius_x_left;
	int center_y = point.y - radius_y_up;

	//�����˹Ȩֵ
	Range x_rng(-(point.x - radius_x_left), radius_x_right - point.x);
	Range y_rng(-(point.y - radius_y_up), radius_y_down - point.y);
	Mat gauss_weight;
	gauss_weight.create(y_rng.end - y_rng.start + 1, x_rng.end - x_rng.start + 1, CV_32FC1);
	for (int i = y_rng.start; i <= y_rng.end; ++i)
	{
		float *ptr_gauss = gauss_weight.ptr<float>(i - y_rng.start);
		for (int j = x_rng.start; j <= x_rng.end; ++j)
			ptr_gauss[j-x_rng.start] = exp((i*i + j*j)*exp_temp);
	}
	
	//������������Χ�������ݶȷ��ȣ��ݶȷ���
	Mat sub_amplit = amplit(Range(radius_y_up, radius_y_down+1), Range(radius_x_left, radius_x_right+1));
	Mat sub_orient = orient(Range(radius_y_up, radius_y_down + 1), Range(radius_x_left, radius_x_right + 1));

	//Mat W = sub_amplit.mul(gauss_weight);//�����˹Ȩ��
	Mat W = sub_amplit;//û�и�˹Ȩ��

	//����ֱ��ͼ
	AutoBuffer<float> buffer(n+4);
	float *temp_hist = buffer + 2;
	for (int i = 0; i < n; ++i)
		temp_hist[i] = 0.f;
	for (int i = 0; i < sub_orient.rows; i++)
	{
		float *ptr_1 = W.ptr<float>(i);
		float *ptr_2 = sub_orient.ptr<float>(i);
		for (int j = 0; j < sub_orient.cols; j++)
		{
			if (((i - center_y)*(i - center_y) + (j - center_x)*(j - center_x)) < radius*radius)
			{
				int bin = cvRound(ptr_2[j] * n / 360.f);
				if (bin>n)
					bin = bin - n;
				if (bin < 0)
					bin = bin + n;
				temp_hist[bin] += ptr_1[j];
			}	
		}
	}

	//ƽ��ֱ��ͼ
	temp_hist[-1] = temp_hist[n - 1];
	temp_hist[-2] = temp_hist[n - 2];
	temp_hist[n] = temp_hist[0];
	temp_hist[n + 1] = temp_hist[1];
	for (int i = 0; i < n; ++i)
	{
		hist[i] = (temp_hist[i - 2] + temp_hist[i + 2])*(1.f / 16.f) +
			(temp_hist[i - 1] + temp_hist[i + 1])*(4.f / 16.f) +
			temp_hist[i] * (6.f / 16.f);
	}

	//���ֱ��ͼ�����ֵ
	float max_value = hist[0];
	for (int i = 1; i < n; ++i)
	{
		if (hist[i]>max_value)
			max_value = hist[i];
	}
	return max_value;
}

/*************************�ú����ڳ߶ȿռ䶨λ������************************/
/*harris_fun��ʾ�߶ȿռ�Harris����
 amplit��ʾ�߶ȿռ������ݶȷ���
 orient��ʾ�߶ȿռ������ݶȽǶ�
 keys��ʾ�߶ȿռ��⵽��������
 */
void Sar_sift::find_space_extrema(const vector<Mat> &harris_fun, const vector<Mat> &amplit, const vector<Mat> &orient, vector<KeyPoint> &keys)
{
	keys.clear();
	int num_rows = harris_fun[0].rows;
	int num_cols = harris_fun[0].cols;
	const int n = SAR_SIFT_ORI_BINS;

	KeyPoint keypoint;
	for (int i = 0; i < Mmax; ++i)
	{
		const Mat &cur_harris_fun = harris_fun[i];
		const Mat &cur_amplit = amplit[i];
		const Mat &cur_orient = orient[i];

		for (int r = SAR_SIFT_BORDER_CONSTANT; r < num_rows - SAR_SIFT_BORDER_CONSTANT; ++r)
		{
			const float *ptr_up = cur_harris_fun.ptr<float>(r-1);
			const float *ptr_cur = cur_harris_fun.ptr<float>(r);
			const float *ptr_nex = cur_harris_fun.ptr<float>(r+1);
			for (int c = SAR_SIFT_BORDER_CONSTANT; c < num_cols - SAR_SIFT_BORDER_CONSTANT; ++c)
			{
				float cur_value = ptr_cur[c];
				if (cur_value>threshold &&
					cur_value>ptr_cur[c - 1] && cur_value > ptr_cur[c + 1] &&
					cur_value > ptr_up[c - 1] && cur_value > ptr_up[c] && cur_value > ptr_up[c + 1] &&
					cur_value > ptr_nex[c - 1] && cur_value > ptr_nex[c] && cur_value > ptr_nex[c + 1])
				{
					float x = c*1.0f; float y = r*1.0f; int layer = i;
					float scale = (float)(sigma*pow(ratio, layer*1.0));
					float hist[n];
					float max_val;
					max_val = calc_orient_hist(amplit[layer], orient[layer], Point2f(x, y), scale, hist, n);

					float mag_thr = max_val*SAR_SIFT_ORI_RATIO;
					for (int k = 0; k < n; ++k)
					{
						int k_left = k <= 0 ? n - 1 : k - 1;
						int k_right = k >= n - 1 ? 0 : k + 1;
						if (hist[k]>mag_thr && hist[k] >= hist[k_left] && hist[k] >= hist[k_right])
						{
							float bin = (float)k + 0.5f*(hist[k_left] - hist[k_right]) / (hist[k_left] + hist[k_right] - 2 * hist[k]);
							if (bin < 0)
								bin = bin + n;
							if (bin >= n)
								bin = bin - n;

							keypoint.pt.x = x*1.0f;//������x��������
							keypoint.pt.y = y*1.0f;//������y��������
							keypoint.size = scale;//������߶�
							keypoint.octave = i;//���������ڲ�
							keypoint.angle = (360.f / n)*bin;//�������������0-360��
							keypoint.response = cur_value;//��������Ӧֵ
							keys.push_back(keypoint);//�����������
						}
					}
				}
			}

		}
	}

}

/*�ú�������matlab�е�meshgrid����*/
/*x_range��ʾx����ķ�Χ
y_range��ʾy����ķ�Χ
X��ʾ���ɵ���x��仯������
Y��ʾ������y��任������
*/
static void meshgrid(const Range &x_range, const Range &y_range, Mat &X, Mat &Y)
{
	int x_start = x_range.start, x_end = x_range.end;
	int y_start = y_range.start, y_end = y_range.end;
	int width = x_end - x_start + 1;
	int height = y_end-y_start+1;

	X.create(height, width, CV_32FC1);
	Y.create(height, width, CV_32FC1);

	for (int i = y_start; i <= y_end; i++)
	{
		float *ptr_1 = X.ptr<float>(i -y_start);
		for (int j = x_start; j <= x_end; ++j)
			ptr_1[j - x_start] = j*1.0f;
	}
	
	for (int i = y_start; i <= y_end; i++)
	{
		float *ptr_2 = Y.ptr<float>(i - y_start);
		for (int j = x_start; j <= x_end; ++j)
			ptr_2[j - x_start] = i*1.0f;
	}	
}


/*************************�ú�������ÿ�������������������*****************************/
/*amplit��ʾ���������ڲ���ݶȷ���ͼ��
 orient��ʾ���������ڲ���ݶȽǶ�ͼ��
 pt��ʾ�������λ��
 scale��ʾ���������ڲ�ĳ߶�
 main_ori��ʾ�������������0-360��
 d��ʾGLOH�Ƕȷ������������Ĭ����8��
 n��ʾÿ�������ڽǶ���0-360��֮��ȷָ�����nĬ����8
 */

/*static void calc_gloh_descriptor(const Mat &amplit, const Mat &orient,Point2f pt, float scale, float main_ori,int d, int n, float *ptr_des)
{
	Point point(cvRound(pt.x), cvRound(pt.y));

	//��������ת�������Һ�����
	float cos_t = cosf(-main_ori / 180.f * (float)CV_PI);
	float sin_t = sinf(-main_ori / 180.f * (float)CV_PI);

	int num_rows = amplit.rows;
	int num_cols = amplit.cols;
	int radius = cvRound(SAR_SIFT_RADIUS_DES*scale);
	radius = min(radius, min(num_rows / 2, num_cols / 2));//����������뾶

	int radius_x_left = point.x - radius;
	int radius_x_right = point.x + radius;
	int radius_y_up = point.y - radius;
	int radius_y_down = point.y + radius;

	//��ֹԽ��
	if (radius_x_left < 0)
		radius_x_left = 0;
	if (radius_x_right>num_cols - 1)
		radius_x_right = num_cols - 1;
	if (radius_y_up < 0)
		radius_y_up = 0;
	if (radius_y_down>num_rows - 1)
		radius_y_down = num_rows - 1;

	//��ʱ��������Χ��������������ģ�����ڸþ���
	int center_x = point.x - radius_x_left;
	int center_y = point.y - radius_y_up;

	//��������Χ�����������ݶȷ��ȣ��ݶȽǶ�
	Mat sub_amplit = amplit(Range(radius_y_up, radius_y_down + 1), Range(radius_x_left, radius_x_right + 1));
	Mat sub_orient = orient(Range(radius_y_up, radius_y_down + 1), Range(radius_x_left, radius_x_right + 1));
	

	//��center_x��center_yλ���ģ�������������������ת
	Range x_rng(-(point.x - radius_x_left), radius_x_right - point.x);
	Range y_rng(-(point.y - radius_y_up), radius_y_down - point.y);
	Mat X, Y;
	meshgrid(x_rng, y_rng, X, Y);
	Mat c_rot = X*cos_t - Y*sin_t;
	Mat r_rot = X*sin_t + Y*cos_t;
	Mat GLOH_angle,GLOH_amplit;
	phase(c_rot, r_rot, GLOH_angle, true);//�Ƕ���0-360��֮��
	GLOH_amplit = c_rot.mul(c_rot) + r_rot.mul(r_rot);//Ϊ�˼ӿ��ٶȣ�û�м��㿪��

	//����Բ�뾶ƽ��
	float R1_pow = (float)radius*radius;//��Բ�뾶ƽ��
	float R2_pow = pow(radius*SAR_SIFT_GLOH_RATIO_R1_R2, 2.f);//�м�Բ�뾶ƽ��
	float R3_pow = pow(radius*SAR_SIFT_GLOH_RATIO_R1_R3, 2.f);//��Բ�뾶ƽ��

	int sub_rows = sub_amplit.rows;
	int sub_cols = sub_amplit.cols;

	//��ʼ����������,�ڽǶȷ���������ӽ��в�ֵ
	int len = (d * 2 + 1)*(n + 1);
	AutoBuffer<float> hist(len);
	for (int i = 0; i < len; ++i)//����
		hist[i] = 0;

	for (int i = 0; i < sub_rows; ++i)
	{
		float *ptr_amplit = sub_amplit.ptr<float>(i);
		float *ptr_orient = sub_orient.ptr<float>(i);
		float *ptr_GLOH_amp = GLOH_amplit.ptr<float>(i);
		float *ptr_GLOH_ang = GLOH_angle.ptr<float>(i);
		for (int j = 0; j < sub_cols; ++j)
		{
			float pix_amplit = ptr_amplit[j];//�����ص��ݶȷ���
			float pix_orient = ptr_orient[j];//�����ص��ݶȷ���
			float pix_GLOH_amp = ptr_GLOH_amp[j];//��������GLOH�����еİ뾶λ��
			float pix_GLOH_ang = ptr_GLOH_ang[j];//��������GLOH�����е�λ�÷���

			int rbin, cbin, obin;
			rbin = pix_GLOH_amp<R3_pow ? 0 : (pix_GLOH_amp>R2_pow ? 2 : 1);//rbin={0,1,2}
			cbin = cvFloor(pix_GLOH_ang*d / 360.f);
			cbin = cbin>=d ? cbin - d : (cbin < 0 ? cbin + d : cbin);//cbin=[0,d-1]

			float o = pix_orient*n/360.f;
			obin = cvFloor(o);
			o = o - obin;//���ݶȽǶȷ�����в�ֵ
			obin = obin >= n ? obin - n : (obin < 0 ? obin + n : obin);//obin=[0,n-1]

			float mag_1 = pix_amplit*(1 - o);
			float mag_2 = pix_amplit - mag_1;
			if (rbin == 0)//��Բ
			{
				hist[obin] += mag_1;
				hist[obin + 1] += mag_2;
			}
			else
			{
				int idx = ((rbin - 1)*d + cbin)*(n + 1) + (n + 1)+obin;
				hist[idx] += mag_1;
				hist[idx + 1] += mag_2;
			}
		}
	}

	//
	hist[0] += hist[n];
	for (int i = 0; i < n; ++i)
		ptr_des[i] = hist[i];
	for (int i = 1; i <= 2; ++i)
	{
		for (int j = 0; j < d; ++j)
		{
			int idx = ((i - 1)*d + j)*(n + 1) + (n + 1);
			hist[idx] += hist[idx + n];
			for (int k = 0; k < n; ++k)
				ptr_des[((i - 1)*d + j)*n + n + k] = hist[idx + k];
		}
	}

	//�������ӽ��й�һ��
	int lenght = (2 * d + 1)*n;
	float norm = 0;
	for (int i = 0; i < lenght; ++i)
	{
		norm = norm + ptr_des[i] * ptr_des[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < lenght; ++i)
	{
		ptr_des[i] = ptr_des[i] / norm;
	}

	//��ֵ�ض�
	for (int i = 0; i < lenght; ++i)
	{
		ptr_des[i] = min(ptr_des[i], DESCR_MAG_THR);
	}

	//�ٴι�һ��
	norm = 0;
	for (int i = 0; i < lenght; ++i)
	{
		norm = norm + ptr_des[i] * ptr_des[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < lenght; ++i)
	{
		ptr_des[i] = ptr_des[i] / norm;
	}

}*/

static void calc_gloh_descriptor(const Mat &amplit, const Mat &orient, Point2f pt, float scale, float main_ori, int d, int n, float *ptr_des)
{
	Point point(cvRound(pt.x), cvRound(pt.y));

	//��������ת�������Һ�����
	float cos_t = cosf(-main_ori / 180.f * (float)CV_PI);
	float sin_t = sinf(-main_ori / 180.f * (float)CV_PI);

	int num_rows = amplit.rows;
	int num_cols = amplit.cols;
	int radius = cvRound(SAR_SIFT_RADIUS_DES*scale);
	radius = min(radius, min(num_rows / 2, num_cols / 2));//����������뾶

	int radius_x_left = point.x - radius;
	int radius_x_right = point.x + radius;
	int radius_y_up = point.y - radius;
	int radius_y_down = point.y + radius;

	//��ֹԽ��
	if (radius_x_left < 0)
		radius_x_left = 0;
	if (radius_x_right>num_cols - 1)
		radius_x_right = num_cols - 1;
	if (radius_y_up < 0)
		radius_y_up = 0;
	if (radius_y_down>num_rows - 1)
		radius_y_down = num_rows - 1;

	//��ʱ��������Χ��������������ģ�����ڸþ���
	int center_x = point.x - radius_x_left;
	int center_y = point.y - radius_y_up;

	//��������Χ�����������ݶȷ��ȣ��ݶȽǶ�
	Mat sub_amplit = amplit(Range(radius_y_up, radius_y_down + 1), Range(radius_x_left, radius_x_right + 1));
	Mat sub_orient = orient(Range(radius_y_up, radius_y_down + 1), Range(radius_x_left, radius_x_right + 1));


	//��center_x��center_yλ���ģ�������������������ת
	Range x_rng(-(point.x - radius_x_left), radius_x_right - point.x);
	Range y_rng(-(point.y - radius_y_up), radius_y_down - point.y);
	Mat X, Y;
	meshgrid(x_rng, y_rng, X, Y);
	Mat c_rot = X*cos_t - Y*sin_t;
	Mat r_rot = X*sin_t + Y*cos_t;
	Mat GLOH_angle, GLOH_amplit;
	phase(c_rot, r_rot, GLOH_angle, true);//�Ƕ���0-360��֮��
	GLOH_amplit = c_rot.mul(c_rot) + r_rot.mul(r_rot);//Ϊ�˼ӿ��ٶȣ�û�м��㿪��

	//����Բ�뾶ƽ��
	float R1_pow = (float)radius*radius;//��Բ�뾶ƽ��
	float R2_pow = pow(radius*SAR_SIFT_GLOH_RATIO_R1_R2, 2.f);//�м�Բ�뾶ƽ��
	float R3_pow = pow(radius*SAR_SIFT_GLOH_RATIO_R1_R3, 2.f);//��Բ�뾶ƽ��

	int sub_rows = sub_amplit.rows;
	int sub_cols = sub_amplit.cols;

	//��ʼ����������,�ڽǶȷ���������ӽ��в�ֵ
	int len = (d * 2 + 1)*n;
	AutoBuffer<float> hist(len);
	for (int i = 0; i < len; ++i)//����
		hist[i] = 0;

	for (int i = 0; i < sub_rows; ++i)
	{
		float *ptr_amplit = sub_amplit.ptr<float>(i);
		float *ptr_orient = sub_orient.ptr<float>(i);
		float *ptr_GLOH_amp = GLOH_amplit.ptr<float>(i);
		float *ptr_GLOH_ang = GLOH_angle.ptr<float>(i);
		for (int j = 0; j < sub_cols; ++j)
		{
			if (((i - center_y)*(i - center_y) + (j - center_x)*(j - center_x)) < radius*radius)
			{
				float pix_amplit = ptr_amplit[j];//�����ص��ݶȷ���
				float pix_orient = ptr_orient[j];//�����ص��ݶȷ���
				float pix_GLOH_amp = ptr_GLOH_amp[j];//��������GLOH�����еİ뾶λ��
				float pix_GLOH_ang = ptr_GLOH_ang[j];//��������GLOH�����е�λ�÷���

				int rbin, cbin, obin;
				rbin = pix_GLOH_amp<R3_pow ? 0 : (pix_GLOH_amp>R2_pow ? 2 : 1);//rbin={0,1,2}
				cbin = cvRound(pix_GLOH_ang*d / 360.f);
				cbin = cbin >d ? cbin - d : (cbin <= 0 ? cbin + d : cbin);//cbin=[1,d]

				obin = cvRound(pix_orient*n / 360.f);
				obin = obin >n ? obin - n : (obin <= 0 ? obin + n : obin);//obin=[1,n]

				if (rbin == 0)//��Բ
					hist[obin - 1] += pix_amplit;
				else
				{
					int idx = ((rbin - 1)*d + cbin-1)*n + n+obin-1;
					hist[idx] += pix_amplit;
				}
			}
		}
	}

	//�������ӽ��й�һ��
	float norm = 0;
	for (int i = 0; i < len; ++i)
	{
		norm = norm + hist[i] * hist[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < len; ++i)
	{
		hist[i] = hist[i] / norm;
	}

	//��ֵ�ض�
	for (int i = 0; i < len; ++i)
	{
		hist[i] = min(hist[i], DESCR_MAG_THR);
	}

	//�ٴι�һ��
	norm = 0;
	for (int i = 0; i < len; ++i)
	{
		norm = norm + hist[i] * hist[i];
	}
	norm = sqrt(norm);
	for (int i = 0; i < len; ++i)
	{
		ptr_des[i] = hist[i] / norm;
	}

}


/***************�ú����������������������������*************/
/*amplit��ʾ�߶ȿռ����ط���
 orient��ʾ�߶ȿռ������ݶȽǶ�
 keys��ʾ��⵽��������
 descriptors��ʾ��������������������M x N��,M��ʾ�����Ӹ�����N��ʾ������ά��
 */
void Sar_sift::calc_descriptors(const vector<Mat> &amplit, const vector<Mat> &orient, const vector<KeyPoint> &keys,Mat &descriptors)
{
	int d = SAR_SIFT_GLOH_ANG_GRID;//d=4����d=8
	int n = SAR_SIFT_DES_ANG_BINS;//n=8Ĭ��

	int num_keys = (int)keys.size();
	int grids = 2*d + 1;
	descriptors.create(num_keys, grids*n,CV_32FC1);

	for (int i = 0; i < num_keys; ++i)
	{
		float *ptr_des = descriptors.ptr<float>(i);
		Point2f point(keys[i].pt);//������λ��
		float scale = keys[i].size;//���������ڲ�ĳ߶�
		int layer = keys[i].octave;//���������ڲ�
		float main_ori = keys[i].angle;//������������

		//����������������������
		calc_gloh_descriptor(amplit[layer], orient[layer], point, scale, main_ori,d, n,ptr_des);
	}
	
}

/***********************�ú���������������************************/
/*image��ʾԭʼͼ��
 keys��ʾ��⵽��������
 amplit��ʾ�߶ȿռ��ݶ�
 orient��ʾ�߶ȿռ�Ƕ�*/
void Sar_sift::detect_keys(const Mat &image, vector<KeyPoint> &keys, vector<Mat> &harris_fun, vector<Mat> &amplit, vector<Mat> &orient)
{
	build_sar_sift_space(image, harris_fun, amplit, orient);

	find_space_extrema(harris_fun, amplit, orient, keys);

	nFeatures = min(nFeatures, SAR_SIFT_MAX_KEYPOINTS);//���ܳ���������������
	if (nFeatures != 0 && nFeatures< (int)keys.size())
	{
		//��������Ӧֵ�Ӵ�С����
		sort(keys.begin(), keys.end(),
			[](const KeyPoint &a, const KeyPoint &b)
		{return a.response>b.response; });

		//ɾ��������������
		keys.erase(keys.begin() + nFeatures, keys.end());
	}
}

/**********************�ú�����������������************************/
/*keys��ʾ��⵽��������
 amplit��ʾ�ݶȷ���ͼ��
 orient��ʾ�ݶȷ���ͼ��
 des��ʾ���ɵ������Ӿ���M x N��M��ʾ�����Ӹ�����N��ʾ������ά��
 */
void Sar_sift::comput_des(const vector<KeyPoint> &keys, const vector<Mat> &amplit, const vector<Mat> &orient, Mat &des)
{
	calc_descriptors(amplit, orient, keys, des);
}