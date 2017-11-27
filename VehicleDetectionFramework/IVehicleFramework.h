#pragma once

#include "string"
#include "memory"

#include "opencv2\opencv.hpp"

struct Vehicle
{
	cv::Rect roi;
	float confidence;
};

enum EPredictionClass
{
	ePD_Background = 0,
	ePD_Aeroplane,
	ePD_Bicycle,
	ePD_Bird,
	ePD_Boat,
	ePD_Bottle,
	ePD_Bus,
	ePD_Car,
	ePD_Cat,
	ePD_Chair,
	ePD_Cow,
	ePD_Diningtable,
	ePD_Dog,
	ePD_Horse,
	ePD_Motorbike,
	ePD_Person,
	ePD_Pottedplant,
	ePD_Sheep,
	ePD_Sofa,
	ePD_Train,
	ePD_TVMonitor
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

