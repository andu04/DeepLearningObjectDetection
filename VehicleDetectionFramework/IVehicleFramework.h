#pragma once

#include "string"
#include "memory"

#include "opencv2\opencv.hpp"

struct Vehicle
{
	cv::Rect roi;
	float confidence;
};

class IVehicleFramework
{
public:
	static std::shared_ptr<IVehicleFramework> Produce();

	virtual ~IVehicleFramework() { };

	virtual bool Init(std::string modelFilePath, std::string trainedParametersFilePath) = 0;
	virtual void Release() = 0;
	virtual void Reset() = 0;

	virtual void ProcessFrame(const cv::Mat& frame) = 0;

	virtual const std::vector<Vehicle>& GetOutputVehicles() = 0;
};

typedef std::shared_ptr<IVehicleFramework> IVehicleFrameworkPtr;

