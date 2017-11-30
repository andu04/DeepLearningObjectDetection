#include "Detector.h"

/*
* ===  FUNCTION  ======================================================================
*         Name:  Detector
*  Description:  Load the model file and weights file
* =====================================================================================
*/
//load modelfile and weights
Detector::Detector(const string& model_file, const string& weights_file)
{
	net_ = boost::shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
	net_->CopyTrainedLayersFrom(weights_file);
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  Detect
*  Description:  Perform detection operation
*                 Warning the max input size should less than 1000*600
* =====================================================================================
*/
bool compare_score(const Info& Info1, const Info& Info2)
{
	return Info1.score > Info2.score;
}

void Detector::Detect(const cv::Mat& frame, vector<Prediction>& detections)
{
	if (frame.empty())
		return;

	cv::Mat cv_new(frame.rows, frame.cols, CV_32FC3, cv::Scalar(0, 0, 0));
	
	for (int h = 0; h < frame.rows; ++h)
	{
		for (int w = 0; w < frame.cols; ++w)
		{
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(frame.at<cv::Vec3b>(cv::Point(w, h))[0]) - float(102.9801);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(frame.at<cv::Vec3b>(cv::Point(w, h))[1]) - float(115.9465);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(frame.at<cv::Vec3b>(cv::Point(w, h))[2]) - float(122.7717);

		}
	}

	cv::Mat cv_resized;
	
	float img_scale = 0.4f;
	float im_info[3];

	int height = int(frame.rows * img_scale);
	int width = int(frame.cols * img_scale);

	cv::resize(cv_new, cv_resized, cv::Size(width, height));
	im_info[0] = cv_resized.rows;
	im_info[1] = cv_resized.cols;
	im_info[2] = img_scale;

	float *data_buf = new float[height * width * 3];
	for (int h = 0; h < height; ++h)
	{
		for (int w = 0; w < width; ++w)
		{
			data_buf[(0 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
			data_buf[(1 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
			data_buf[(2 * height + h)*width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
		}
	}

	net_->blob_by_name("data")->Reshape(1, 3, height, width);
	net_->blob_by_name("data")->set_cpu_data(data_buf);
	net_->blob_by_name("im_info")->set_cpu_data(im_info);

	net_->ForwardFrom(0);

	const float* bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	const float* pred_cls = net_->blob_by_name("cls_prob")->cpu_data();
	const float* rois = net_->blob_by_name("rois")->cpu_data();
	
	predictionsCount = net_->blob_by_name("rois")->num();

	bbox_transform_inv(bbox_delt, pred_cls, rois, img_scale, frame.rows, frame.cols);

	GetResults(detections);
}

void Detector::GetResults(vector<Prediction>& detections)
{
	predictionsPerClass.resize(predictionsCount * 5);
	for (int i = 0; i < CLASS_NUM; i++)
	{
		for (int j = 0; j < predictionsCount; j++)
		{
			for (int k = 0; k < 5; k++)
			{
				predictionsPerClass[j * 5 + k] = predictions[(i * predictionsCount + j) * 5 + k];
			}
		}

		vector<vector<float> > pred_boxes;
		vector<float> confidence;
		for (int j = 0; j < predictionsCount; j++)
		{
			vector<float> tmp_box;
			tmp_box.push_back(predictionsPerClass[j * 5 + 0]);
			tmp_box.push_back(predictionsPerClass[j * 5 + 1]);
			tmp_box.push_back(predictionsPerClass[j * 5 + 2]);
			tmp_box.push_back(predictionsPerClass[j * 5 + 3]);
			pred_boxes.push_back(tmp_box);
			confidence.push_back(predictionsPerClass[j * 5 + 4]);
		}

		if (!pred_boxes.empty() && !confidence.empty())
		{
			apply_nms(pred_boxes, confidence, NMS_THRESH);

			for (int t = 0; t < pred_boxes.size(); t++)
			{
				if (confidence[t] > CONF_THRESH)
				{
					Prediction p;
					p.classID = i;
					p.confidence = confidence[t];
					p.roi = cv::Rect(cv::Point(pred_boxes[t][0], pred_boxes[t][1]), cv::Point(pred_boxes[t][2], pred_boxes[t][3]));

					detections.push_back(p);
				}
			}
		}
	}
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  boxes_sort
*  Description:  Sort the bounding box according score
* =====================================================================================
*/
//Using for box sort
void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
	vector<Info> my;
	Info tmp;
	for (int i = 0; i< num; i++)
	{
		tmp.score = pred[i * 5 + 4];
		tmp.head = pred + i * 5;
		my.push_back(tmp);
	}
	std::sort(my.begin(), my.end(), compare_score);
	for (int i = 0; i<num; i++)
	{
		for (int j = 0; j<5; j++)
			sorted_pred[i * 5 + j] = my[i].head[j];
	}
}

/*
* ===  FUNCTION  ======================================================================
*         Name:  bbox_transform_inv
*  Description:  Compute bounding box regression value
* =====================================================================================
*/
void Detector::bbox_transform_inv(const float* box_deltas, const float* pred_cls, const float* rois, float img_scale, int img_height, int img_width)
{
	predictions.resize(predictionsCount * 5 * CLASS_NUM);

	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	for (int i = 0; i< predictionsCount; i++)
	{
		width = rois[i * 5 + 3] / img_scale - rois[i * 5 + 1] / img_scale + 1.0;
		height = rois[i * 5 + 4] / img_scale - rois[i * 5 + 2] / img_scale + 1.0;
		ctr_x = rois[i * 5 + 1] / img_scale + 0.5 * width;
		ctr_y = rois[i * 5 + 2] / img_scale + 0.5 * height;
		for (int j = 1; j< CLASS_NUM; j++)
		{
			dx = box_deltas[(i*CLASS_NUM + j) * 4 + 0];
			dy = box_deltas[(i*CLASS_NUM + j) * 4 + 1];
			dw = box_deltas[(i*CLASS_NUM + j) * 4 + 2];
			dh = box_deltas[(i*CLASS_NUM + j) * 4 + 3];
			pred_ctr_x = ctr_x + width*dx;
			pred_ctr_y = ctr_y + height*dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);

			predictions[(j*predictionsCount + i) * 5 + 0] = max(min(pred_ctr_x - 0.5* pred_w, img_width - 1), 0);
			predictions[(j*predictionsCount + i) * 5 + 1] = max(min(pred_ctr_y - 0.5* pred_h, img_height - 1), 0);
			predictions[(j*predictionsCount + i) * 5 + 2] = max(min(pred_ctr_x + 0.5* pred_w, img_width - 1), 0);
			predictions[(j*predictionsCount + i) * 5 + 3] = max(min(pred_ctr_y + 0.5* pred_h, img_height - 1), 0);
			predictions[(j*predictionsCount + i) * 5 + 4] = pred_cls[i*CLASS_NUM + j];
		}
	}
}

void Detector::apply_nms(vector<vector<float> > &pred_boxes, vector<float> &confidence, float nms_threshold)
{
	for (int i = 0; i < pred_boxes.size() - 1; i++)
	{
		float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1) *(pred_boxes[i][3] - pred_boxes[i][1] + 1);
		for (int j = i + 1; j < pred_boxes.size(); j++)
		{
			float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1) *(pred_boxes[j][3] - pred_boxes[j][1] + 1);

			float x1 = max(pred_boxes[i][0], pred_boxes[j][0]);
			float y1 = max(pred_boxes[i][1], pred_boxes[j][1]);
			float x2 = min(pred_boxes[i][2], pred_boxes[j][2]);
			float y2 = min(pred_boxes[i][3], pred_boxes[j][3]);

			float width = x2 - x1;
			float height = y2 - y1;
			if (width > 0 && height > 0)
			{
				float IOU = width * height / (s1 + s2 - width * height);
				if (IOU > nms_threshold)
				{
					if (confidence[i] >= confidence[j])
					{
						pred_boxes.erase(pred_boxes.begin() + j);
						confidence.erase(confidence.begin() + j);
						j--;
					}
					else
					{
						pred_boxes.erase(pred_boxes.begin() + i);
						confidence.erase(confidence.begin() + i);
						i--;
						break;
					}
				}
			}
		}
	}
}