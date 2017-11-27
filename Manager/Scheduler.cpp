#include "Scheduler.h"

#define TIME_UNIT 0.03333f
#define MILLISECONDS_PER_SECOND 1000

Scheduler::Scheduler()
{
	m_pVideoDecoder = std::shared_ptr<IVideoDecoder>();
	m_pVehicleFramework = std::shared_ptr<IVehicleFramework>();
}


Scheduler::~Scheduler()
{
	m_frame.release();
}

bool Scheduler::Init(std::string path, bool fromFolder)
{
	if (m_pVideoDecoder == nullptr)
		m_pVideoDecoder = IVideoDecoder::Produce(fromFolder ? eDT_Folder : eDT_Video);

	if (m_pVehicleFramework == nullptr)
		m_pVehicleFramework = IVehicleFramework::Produce();


	if (!m_pVideoDecoder->Load(path, eFF_BGR))
		return false;

	int channels = 0;
	m_pVideoDecoder->GetFrameSize(m_frameSize.width, m_frameSize.height, channels);

	m_frame = cv::Mat(m_frameSize, CV_8UC3);

	std::string model_file("D:\\Faster R-CNN Additional Files\\model\\ZF\\test.prototxt");
	std::string trained_file("D:\\Faster R-CNN Additional Files\\model\\ZF\\ZF_faster_rcnn_final.caffemodel");

	if (!m_pVehicleFramework->Init(model_file, trained_file))
	{
		m_pVehicleFramework->Release();
		return false;
	}

	return true;
}

void Scheduler::Execute()
{
	bool isPaused = false;
	bool showNextFrame = false;
	int keyPressed = -1;

	int frameSizeBytes = m_frameSize.width * m_frameSize.height * m_frame.channels();
	m_pVideoDecoder->GetFrame(135);
	byte* framePtr = m_pVideoDecoder->GetNextFrame(); 
	if (framePtr == nullptr || frameSizeBytes == 0)
		return;

	while (framePtr != nullptr)
	{
		memcpy(m_frame.data, framePtr, frameSizeBytes);

		if (!isPaused || showNextFrame)
		{
			double t0 = (double)cv::getTickCount();
			m_pVehicleFramework->ProcessFrame(m_frame);
			t0 = ((double)cv::getTickCount() - t0) * 1000.f / cv::getTickFrequency();
			std::cout << "Time to process frame: " << t0 << "\n";
			Render();
		}

		keyPressed = cv::waitKey(TIME_UNIT * MILLISECONDS_PER_SECOND);

		switch (keyPressed)
		{
		case eKP_ESC:
			isPaused = true;
			showNextFrame = false;
			framePtr = nullptr;
			break;
		case eKP_SPACE:
			isPaused = !isPaused;			
			showNextFrame = false;
			break;
		case eKP_RIGHT_ARROW:
			isPaused = true;
			showNextFrame = true;
			break;
		default:
			showNextFrame = false;
			break;
		}
			
		framePtr = m_pVideoDecoder->GetNextFrame();
	}
}

void Scheduler::Render()
{
	auto currentFrameDetections = m_pVehicleFramework->GetOutputVehicles();
	for (const auto& vehicle : currentFrameDetections)
		cv::rectangle(m_frame, vehicle.roi, cv::Scalar(0, 0, 255), 2);

	cv::imshow("Movie", m_frame);
}