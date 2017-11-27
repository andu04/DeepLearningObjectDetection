#pragma once

#include "IVideoDecoder.h"
#include "IVehicleFramework.h"

#include "opencv2\opencv.hpp"

enum EKeyPressed
{
	eKP_NO_KEY_PRESSED = -1,
	eKP_ESC = 27,
	eKP_SPACE = 32,
	eKP_LEFT_ARROW = 2424832,
	eKP_RIGHT_ARROW = 2555904
};

class Scheduler
{
public:
	Scheduler();
	~Scheduler();

	bool Init(std::string path, bool fromFolder);
	void Execute();
	void Render();

private:

	IVideoDecoderPtr m_pVideoDecoder;
	IVehicleFrameworkPtr m_pVehicleFramework;

	cv::Mat m_frame;
	cv::Size m_frameSize;
};

