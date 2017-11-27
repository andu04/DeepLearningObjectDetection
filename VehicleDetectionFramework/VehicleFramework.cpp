#include "VehicleFramework.h"



VehicleFramework::VehicleFramework()
{
	m_pVehicleDetector = std::shared_ptr<Detector>();
}


VehicleFramework::~VehicleFramework()
{
	Release();
}


bool VehicleFramework::Init(std::string modelFilePath, std::string trainedParametersFilePath)
{
	m_pVehicleDetector = std::shared_ptr<Detector>(new Detector(modelFilePath, trainedParametersFilePath));

	if (m_pVehicleDetector == nullptr)
		return false;

	return true;
}

void VehicleFramework::Release()
{
	Reset();

	m_pVehicleDetector.reset();
}
void VehicleFramework::Reset()
{
	m_vehicleCandidates.clear();
	m_outputVehicles.clear();
}

void VehicleFramework::ProcessFrame(const cv::Mat& frame)
{
	m_searchRoi.x = frame.cols / 3;
	m_searchRoi.y = frame.rows / 4;
	m_searchRoi.width = frame.cols / 2;
	m_searchRoi.height = frame.rows * 3 / 4;
	m_searchRoi = m_searchRoi & cv::Rect(0, 0, frame.cols, frame.rows);

	m_originalFrameSize = cv::Size(frame.cols, frame.rows);

	DetectVehicles(frame(m_searchRoi));
	SetOutputVehicles();
}

const std::vector<Vehicle>& VehicleFramework::GetOutputVehicles()
{
	return m_outputVehicles;
}

void VehicleFramework::DetectVehicles(const cv::Mat& frame)
{
	std::vector<Prediction> newDetections;
	m_pVehicleDetector->Detect(frame, newDetections);

	m_vehicleCandidates.clear();

	for (const auto& prediction : newDetections)
	{
		bool isVehicle = prediction.classID == ePD_Bicycle || prediction.classID == ePD_Bus || 
						 prediction.classID == ePD_Car || prediction.classID == ePD_Motorbike;
		if (prediction.confidence > 0.95f && isVehicle)
		{
			cv::Rect predRoi = prediction.roi;
			predRoi.x += m_searchRoi.x;
			predRoi.y += m_searchRoi.y;
			predRoi = predRoi & cv::Rect(cv::Size(0, 0), m_originalFrameSize);

			if (predRoi.width <= 5 || predRoi.height <= 5)
				continue;

			VehicleCandidate newCandidate;
			newCandidate.lastDetection = predRoi;
			newCandidate.detections.push_back(predRoi);
			newCandidate.overallConfidence = prediction.confidence;

			m_vehicleCandidates.push_back(newCandidate);
		}
	}
}

void VehicleFramework::SetOutputVehicles()
{
	m_outputVehicles.clear();

	for (const auto& detectedVehicle : m_vehicleCandidates)
	{
		Vehicle newVehicle;
		newVehicle.roi = detectedVehicle.lastDetection;
		newVehicle.confidence = detectedVehicle.overallConfidence;

		m_outputVehicles.push_back(newVehicle);
	}
}