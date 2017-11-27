#include "VideoDecoder.h"
#include "FolderDecoder.h"

std::shared_ptr<IVideoDecoder> IVideoDecoder::Produce(EDecoderType eDetectorType)
{
	switch (eDetectorType)
	{
	case eDT_Video:
		return std::shared_ptr<IVideoDecoder>(new VideoDecoder());
	case eDT_Folder:
		return std::shared_ptr<IVideoDecoder>(new FolderDecoder());
	default:
		return nullptr;
		break;
	}
	
}