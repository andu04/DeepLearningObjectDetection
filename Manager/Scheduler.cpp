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

bool Scheduler::Init(std::string path)
{
	if (m_pVideoDecoder == nullptr)
		m_pVideoDecoder = IVideoDecoder::Produce();

	if (m_pVehicleFramework == nullptr)
		m_pVehicleFramework = IVehicleFramework::Produce();


	if (!m_pVideoDecoder->Load(path, eFF_BGR))
		return false;

	int channels = 0;
	m_pVideoDecoder->GetFrameSize(m_frameSize.width, m_frameSize.height, channels);

	m_frame = cv::Mat(m_frameSize, CV_8UC3);

	return true;
}

void Scheduler::Execute()
{
	bool isPaused = false;
	bool showNextFrame = false;
	int keyPressed = -1;

	int frameSizeBytes = m_frameSize.width * m_frameSize.height * m_frame.channels();

	byte* framePtr = m_pVideoDecoder->GetNextFrame(); 
	if (framePtr == nullptr || frameSizeBytes == 0)
		return;

	memcpy(m_frame.data, framePtr, frameSizeBytes);

	while (framePtr != nullptr)
	{
		cv::imshow("Movie", m_frame);
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
			
		if (!isPaused || showNextFrame)
		{
			framePtr = m_pVideoDecoder->GetNextFrame();
			if (framePtr != nullptr)
				memcpy(m_frame.data, framePtr, frameSizeBytes);
		}
	}
}