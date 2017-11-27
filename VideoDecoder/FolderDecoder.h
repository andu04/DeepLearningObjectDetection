#pragma once
#include "IVideoDecoder.h"

#include "opencv2/opencv.hpp"

class FolderDecoder : public IVideoDecoder
{
public:
	FolderDecoder();
	~FolderDecoder();

	virtual bool Load(std::string path, EFrameFormat decodedFrameFormat) override;

	virtual int GetFramesCount() override;
	virtual void GetFrameSize(int& width, int& height, int& channels) override;
	virtual byte* GetFrame(int frameIndex) override;
	virtual byte* GetNextFrame() override;
	virtual int GetFPS() override;

private:
	bool JumpToFrame(int frameIndex);
	void ConvertFrameColor();

	std::string m_path;

	int m_framesCount;
	int m_frameWidth;
	int m_frameHeight;
	int m_frameChannels;
	int m_FPS;
	EFrameFormat m_decodedFrameFormat;

	cv::Mat m_retrievedFrame;
	cv::Mat m_decodedFrame;

	std::string m_currentFramePath;
	int m_currentFrameIndex;
};

