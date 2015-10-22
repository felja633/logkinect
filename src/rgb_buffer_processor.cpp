#include "rgb_buffer_processor.h"
#include <iostream>

namespace logkinect
{


static THREAD_RET_VAL kuk(THREAD_ARG_TYPE this_is_this)
{
	RgbBufferProcessor* my_this = reinterpret_cast<RgbBufferProcessor*> (this_is_this);
	my_this->bufferDecompress();
}


RgbBufferProcessor::RgbBufferProcessor()
{
  decompressor = tjInitDecompress();
  if(decompressor == 0)
  {
    std::cerr << "[RgbPacketProcessor] Failed to initialize TurboJPEG decompressor! TurboJPEG error: '" << tjGetErrorStr() << "'" << std::endl;
  }

  timing_acc = 0.0;
  timing_acc_n = 0.0;
  timing_current_start = 0.0;

	//tex = texture.create2DTextureByte(1920,1080,mProcessed_data,GL_RGB,GL_RGB);
	mRelease = false;
}

RgbBufferProcessor::~RgbBufferProcessor()
{
	stop();
  if(decompressor != 0)
  {
    if(tjDestroy(decompressor) == -1)
    {
      std::cerr << "[RgbPacketProcessor] Failed to destroy TurboJPEG decompressor! TurboJPEG error: '" << tjGetErrorStr() << "'" << std::endl;
    }
  }
}

void RgbBufferProcessor::start()
{
	mStop = false;
	mThread.spawn(&kuk, (THREAD_ARG_TYPE)this);
}

void RgbBufferProcessor::stop()
{
	if(mStop)
		return;

	mStop = true;
	mRgbLockCv.notifyOne();
	std::cout<<"RgbBufferProcessor wait for join..\n";
	mThread.join();
}

/*std::thread RgbBufferProcessor::spawn()
{
	return std::thread( [this] { this->bufferDecompress(); } );
}*/

void RgbBufferProcessor::process(unsigned char* buffer, int length)
{
#ifdef WIN32
	mRgbMutex.lock();
#else
#if __cplusplus > 199711L
	std::unique_lock<std::mutex> rgb_lck = mRgbMutex.lock();
#else

	mRgbMutex.lock();
#endif
#endif

		while(mRelease)
		{

#ifdef WIN32
			mRgbLockCv.wait(&mRgbMutex);
#else
#if __cplusplus > 199711L
			mRgbLockCv.wait(rgb_lck);
#else
			mRgbLockCv.wait(&mRgbMutex);
#endif
#endif
		}

	mLength = length;
	mJpeg_buffer = buffer;

	mRelease = true;
	mRgbLockCv.notifyOne();
#ifdef WIN32
	mRgbMutex.unlock();
#else
#if __cplusplus > 199711L
	mRgbMutex.unlock(rgb_lck);
#else
	mRgbMutex.unlock();
#endif
#endif
}

void RgbBufferProcessor::readResult(logkinect::Color_Packet& packet)
{
#ifdef WIN32
	mRgbMutex.lock();
#else
#if __cplusplus > 199711L
	std::unique_lock<std::mutex> rgb_lck = mRgbMutex.lock();
#else
	mRgbMutex.lock();
#endif
#endif
	//std::cout<<"update texture \n";
	//texture.update2DTextureByte(1920,1080,mProcessed_data,GL_RGB,tex);
	packet.buffer = mProcessed_data;
	packet.height = 1080;
	packet.width = 1920;
	//wait for upload
#ifdef WIN32
	mRgbMutex.unlock();
#else
#if __cplusplus > 199711L
	mRgbMutex.unlock(rgb_lck);
#else
	mRgbMutex.unlock();
#endif
#endif
}

void RgbBufferProcessor::bufferDecompress()
{

	while(!mStop)
	{

		//protect buffers
#ifdef WIN32 
		mRgbMutex.lock();
#else
#if __cplusplus > 199711L
		std::unique_lock<std::mutex> rgb_lck = mRgbMutex.lock();
#else
		mRgbMutex.lock();
#endif
#endif
		
		//wait for rendering thread to catch up
		while(!mRelease & !mStop)
		{
			
#ifdef WIN32
			mRgbLockCv.wait(&mRgbMutex);
#else
#if __cplusplus > 199711L
			mRgbLockCv.wait(rgb_lck);
#else
			mRgbLockCv.wait(&mRgbMutex);
#endif
#endif
		}

		if(mStop)
		{
			//delete[] mJpeg_buffer;
			return;
		}
		
		if(decompressor != 0)
		{
			//std::cout<<"before decompress \n";
		  int r = tjDecompress2(decompressor, mJpeg_buffer, mLength, mProcessed_data, 1920, 1920 * tjPixelSize[TJPF_RGB], 1080, TJPF_RGB, 0);

		  if(r != 0)
		  {
		    std::cerr << "[RgbPacketProcessor::doProcess] Failed to decompress rgb image! TurboJPEG error: '" << tjGetErrorStr() << "'" << std::endl;
		  }

		}
		
		
		delete[] mJpeg_buffer;
		mRelease = false;
		mRgbLockCv.notifyOne();
#ifdef WIN32
		mRgbMutex.unlock();
#else
#if __cplusplus > 199711L
		mRgbMutex.unlock(rgb_lck);
#else
		mRgbMutex.unlock();
#endif
#endif
	}
}

}


