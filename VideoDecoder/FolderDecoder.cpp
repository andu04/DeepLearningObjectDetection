#include "FolderDecoder.h"



FolderDecoder::FolderDecoder()
{
	m_framesCount = 0;
	m_frameWidth = 0;
	m_frameHeight = 0;
	m_frameChannels = 0;

	m_currentFrameIndex = -1;
}


FolderDecoder::~FolderDecoder()
{
	m_retrievedFrame.release();
	m_decodedFrame.release();

	m_framesCount = 0;
	m_frameWidth = 0;
	m_frameHeight = 0;
	m_frameChannels = 0;

	m_currentFrameIndex = -1;
}

bool FolderDecoder::Load(std::string path, EFrameFormat decodedFrameFormat)
{
	m_path = path;

	m_framesCount = 899;

	std::stringstream ss;
	ss << path << "\\I1_000000.png";

	cv::Mat temp = cv::imread(ss.str());
	m_frameWidth = temp.cols;
	m_frameHeight = temp.rows;
	temp.release();

	switch (decodedFrameFormat)
	{
	case eFF_Grayscale:
		m_frameChannels = 1;
		break;
	case eFF_BGR:
		m_frameChannels = 3;
		break;
	default:
		break;
	}

	m_decodedFrameFormat = decodedFrameFormat;

	m_FPS = 0;

	m_currentFrameIndex = -1;

	return true;
}

int FolderDecoder::GetFramesCount()
{
	return m_framesCount;
}

void FolderDecoder::GetFrameSize(int& width, int& height, int& channels)
{
	width = m_frameWidth;
	height = m_frameHeight;
	channels = m_frameChannels;
}

int FolderDecoder::GetFPS()
{
	return m_FPS;
}

byte* FolderDecoder::GetFrame(int frameIndex)
{
	if (!JumpToFrame(frameIndex))
		return nullptr;

	m_retrievedFrame = cv::imread(m_currentFramePath);

	if (m_retrievedFrame.empty())
		return nullptr;

	m_decodedFrame.release();

	if (m_retrievedFrame.channels() != m_frameChannels)
		ConvertFrameColor();
	else
		m_decodedFrame = m_retrievedFrame.clone();

	m_retrievedFrame.release();

	return m_decodedFrame.data;
}

byte* FolderDecoder::GetNextFrame()
{
	return GetFrame(m_currentFrameIndex + 1);
}

bool FolderDecoder::JumpToFrame(int frameIndex)
{
	if (m_framesCount <= frameIndex)
		return false;

	char *frameNumerStr = new char[7];
	std::sprintf(frameNumerStr, "%06d", frameIndex);
	std::stringstream ss;
	ss << m_path << "\\I1_" << frameNumerStr << ".png";
	m_currentFramePath = ss.str();

	m_currentFrameIndex = frameIndex;
	return true;
}

void FolderDecoder::ConvertFrameColor()
{
	int retrievedChannels = m_retrievedFrame.channels();
	if (retrievedChannels == 3 && m_frameChannels == 1)
		cv::cvtColor(m_retrievedFrame, m_decodedFrame, CV_BGR2GRAY);
	else if (retrievedChannels == 1 && m_frameChannels == 3)
		cv::cvtColor(m_retrievedFrame, m_decodedFrame, CV_GRAY2BGR);
}
