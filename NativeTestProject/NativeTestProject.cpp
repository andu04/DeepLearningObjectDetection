// NativeTestProject.cpp : Defines the entry point for the console application.
//

//#include "Detector.h"

#include "Scheduler.h"

//using namespace caffe;

int main()
{
	//std::string model_file("D:\\Faster R-CNN Additional Files\\model\\ZF\\test.prototxt");
	//std::string trained_file("D:\\Faster R-CNN Additional Files\\model\\ZF\\ZF_faster_rcnn_final.caffemodel");
	//
	//int GPUID = 0;
	//Caffe::SetDevice(GPUID);
	//Caffe::set_mode(Caffe::GPU);

	////Caffe::set_mode(Caffe::CPU);
	//Detector det = Detector(model_file, trained_file);
	//
	//string im_names="C:\\Users\\Andi\\Desktop\\test.jpg";
	//det.Detect(im_names);
	
	auto scheduler = new Scheduler();

	std::string videoPath = "C:\\Users\\Andi\\Pictures\\Camera Roll\\WIN_20171115_22_01_48_Pro.mp4";
	if (!scheduler->Init(videoPath))
		return -1;

	scheduler->Execute();

	delete scheduler;
	return 0;
}

