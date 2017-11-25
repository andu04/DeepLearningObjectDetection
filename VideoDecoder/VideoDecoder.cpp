#include "VideoDecoder.h"



VideoDecoder::VideoDecoder()
{
	m_videoCapture = cv::VideoCapture();
	
	m_framesCount = 0;
	m_frameWidth = 0;
	m_frameHeight = 0;
	m_frameChannels = 0;

	m_currentFrameIndex = -1;
}


VideoDecoder::~VideoDecoder()
{
	m_retrievedFrame.release();
	m_decodedFrame.release();
	m_videoCapture.release();

	m_framesCount = 0;
	m_frameWidth = 0;
	m_frameHeight = 0;
	m_frameChannels = 0;

	m_currentFrameIndex = -1;
}


bool VideoDecoder::Load(std::string path, EFrameFormat decodedFrameFormat)
{
	m_path = path;

	if (m_videoCapture.isOpened())
		m_videoCapture.release();

	if (!m_videoCapture.open(path))
		return false;

	m_framesCount = static_cast<int>(m_videoCapture.get(CV_CAP_PROP_FRAME_COUNT));
	m_frameWidth = static_cast<int>(m_videoCapture.get(CV_CAP_PROP_FRAME_WIDTH));
	m_frameHeight = static_cast<int>(m_videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT));

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

	m_FPS = static_cast<int>(m_videoCapture.get(CV_CAP_PROP_FPS));

	m_currentFrameIndex = -1;
}

int VideoDecoder::GetFramesCount()
{
	return m_framesCount;
}

void VideoDecoder::GetFrameSize(int& width, int& height, int& channels)
{
	width = m_frameWidth;
	height = m_frameHeight;
	channels = m_frameChannels;
}

int VideoDecoder::GetFPS()
{
	return m_FPS;
}

byte* VideoDecoder::GetFrame(int frameIndex)
{
	if (!m_videoCapture.isOpened() || m_framesCount == 0)
		return nullptr;

	if (!JumpToFrame(frameIndex))
		return nullptr;

	if (!m_videoCapture.retrieve(m_retrievedFrame))
		return nullptr;

	m_decodedFrame.release();

	if (m_retrievedFrame.channels() != m_frameChannels)
		ConvertFrameColor();
	else
		m_decodedFrame = m_retrievedFrame.clone();

	m_retrievedFrame.release();

	return m_decodedFrame.data;
}

byte* VideoDecoder::GetNextFrame()
{
	return GetFrame(m_currentFrameIndex + 1);
}

bool VideoDecoder::JumpToFrame(int frameIndex)
{
	if (m_framesCount <= frameIndex)
		return false;

	bool result = false;
	if (frameIndex == m_currentFrameIndex + 1)
		result = m_videoCapture.grab();
	else
	{
		m_videoCapture.set(CV_CAP_PROP_POS_FRAMES, frameIndex);
		result = m_videoCapture.grab();

		while (result && m_videoCapture.get(CV_CAP_PROP_POS_FRAMES) != frameIndex)
			 result = m_videoCapture.grab();
	}

	m_currentFrameIndex = frameIndex;
	return result;
}

void VideoDecoder::ConvertFrameColor()
{
	int retrievedChannels = m_retrievedFrame.channels();
	if (retrievedChannels == 3 && m_frameChannels == 1)
		cv::cvtColor(m_retrievedFrame, m_decodedFrame, CV_BGR2GRAY);
	else if (retrievedChannels == 1 && m_frameChannels == 3)
		cv::cvtColor(m_retrievedFrame, m_decodedFrame, CV_GRAY2BGR);
}