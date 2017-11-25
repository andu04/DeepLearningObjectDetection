#include "VideoDecoder.h"

std::shared_ptr<IVideoDecoder> IVideoDecoder::Produce()
{
	return std::shared_ptr<IVideoDecoder>(new VideoDecoder());
}