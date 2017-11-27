#pragma once
#include "IVehicleFramework.h"
#include "Detector.h"

struct VehicleCandidate
{
	std::vector<cv::Rect> detections;
	cv::Rect lastDetection;

	float overallConfidence;

	VehicleCandidate()
	{
		detections.clear();
		overallConfidence = .0f;
	}
};

class VehicleFramework : public IVehicleFramework
{
public:
	VehicleFramework();
	~VehicleFramework();
	
	virtual bool Init(std::string modelFilePath, std::string trainedParametersFilePath) override;
	virtual void Release() override;
	virtual void Reset() override;

	virtual void ProcessFrame(const cv::Mat& frame) override;

	virtual const std::vector<Vehicle>& GetOutputVehicles() override;


private:
	void DetectVehicles(const cv::Mat& frame);
	void SetOutputVehicles();


	std::shared_ptr<Detector> m_pVehicleDetector;
	std::vector<VehicleCandidate> m_vehicleCandidates;
	std::vector<Vehicle> m_outputVehicles;

	cv::Rect m_searchRoi;
	cv::Size m_originalFrameSize;
};

