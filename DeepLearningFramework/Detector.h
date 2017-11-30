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

#define CONF_THRESH 0.95
#define NMS_THRESH 0.5


struct Prediction
{
	cv::Rect roi;
	float confidence;
	int classID;
};
/*
* ===  Class  ======================================================================
*         Name:  Detector
*  Description:  FasterRCNN CXX Detector
* =====================================================================================
*/

class Detector {
public:
	Detector(const string& model_file, const string& weights_file);

	void Detect(const cv::Mat& image, vector<Prediction>& detections);
	
	void bbox_transform_inv(const float* box_deltas, const float* pred_cls, const float* rois, float img_scale, int img_height, int img_width);
	void boxes_sort(int num, const float* pred, float* sorted_pred);
	void apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence, float nms_threshold);

	~Detector() { }
private:
	
	Detector() {}
	void GetResults(vector<Prediction>& detections);

	boost::shared_ptr<Net<float> > net_;
	

	
	/*float* pred_per_class;
	float* pred;
	int num;*/

	vector<float> predictions;
	vector<float> predictionsPerClass;
	int predictionsCount;

	std::vector<Prediction> m_predictions;

	cv::Mat cv_img;
};

struct Info
{
	float score;
	const float* head = NULL;
};

