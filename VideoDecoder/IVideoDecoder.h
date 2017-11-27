#pragma once

#include "string"
#include "memory"

typedef unsigned char byte;

enum EFrameFormat
{
	eFF_Grayscale,
	eFF_BGR
};

enum EDecoderType
{
	eDT_Video,
	eDT_Folder
};

class IVideoDecoder
{
public:
	
	static std::shared_ptr<IVideoDecoder> Produce(EDecoderType eDetectorType);

	virtual bool Load(std::string path, EFrameFormat decodedFrameFormat) = 0;
	
	virtual int GetFramesCount() = 0;
	virtual void GetFrameSize(int& width, int& height, int& channels) = 0;
	virtual byte* GetFrame(int frameIndex) = 0;
	virtual byte* GetNextFrame() = 0;
	virtual int GetFPS() = 0;

	virtual ~IVideoDecoder() { }
};

typedef std::shared_ptr<IVideoDecoder> IVideoDecoderPtr;