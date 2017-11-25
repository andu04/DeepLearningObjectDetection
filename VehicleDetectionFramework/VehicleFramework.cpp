#include "VehicleFramework.h"



VehicleFramework::VehicleFramework()
{
}


VehicleFramework::~VehicleFramework()
{
}


bool VehicleFramework::Init(std::string modelFilePath, std::string trainedParametersFilePath)
{

}
void VehicleFramework::Release()
{

}
void VehicleFramework::Reset()
{

}

void VehicleFramework::ProcessFrame(const cv::Mat& frame)
{
	DetectVehicles(frame);
	SetOutputVehicles();
}

const std::vector<Vehicle>& VehicleFramework::GetOutputVehicles()
{
	return m_outputVehicles;
}

void VehicleFramework::DetectVehicles(const cv::Mat& frame)
{

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