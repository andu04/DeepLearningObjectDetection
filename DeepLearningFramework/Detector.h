#pragma once
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>

using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a) :(b))
#define min(a, b) (((a)<(b)) ? (a) :(b))

//background and car
#define CLASS_NUM 21

#define CONF_THRESH 0.99
#define NMS_THRESH 0.5

/*
* ===  Class  ======================================================================
*         Name:  Detector
*  Description:  FasterRCNN CXX Detector
* =====================================================================================
*/

class Detector {
public:
	Detector(const string& model_file, const string& weights_file);

	void Detect(const cv::Mat& image);
	void Detect(std::string im_name);

	void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
	void vis_detections(cv::Mat& image, vector<vector<float> > pred_boxes, vector<float> confidence, float conf_thresh, cv::Scalar color);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
	void apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence, float nms_threshold);

	void getResults(cv::Mat& img, float nms_threshold, float confidence);

	~Detector()
	{
		if (pred_per_class != nullptr)
		{
			delete[] pred_per_class;
			delete[] pred;
		}
	}
private:
	boost::shared_ptr<Net<float> > net_;
	Detector() { pred_per_class = nullptr; }

	
	float* pred_per_class;
	float* pred;
	int num;
	cv::Mat cv_img;
};

struct Info
{
	float score;
	const float* head = NULL;
};

