
#ifndef RGB_BUFFER_PROCESSOR_H
#define RGB_BUFFER_PROCESSOR_H

#include <turbojpeg.h>
#include "farsan_thread.h"
//#include "GLHelper.h"

namespace logkinect
{

struct Color_Packet
{
	unsigned char* buffer;
	size_t length, bytes_per_element, height, width;
};



class RgbBufferProcessor
{
public:

  tjhandle decompressor;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;
	
	unsigned char* mJpeg_buffer;
	unsigned char mProcessed_data[1080*1920*3];
	unsigned int mLength;
	//logkinect::Color_Packet packet_;
	bool mRelease;
  RgbBufferProcessor();
  ~RgbBufferProcessor();
	void start();
	void stop();
  void process(unsigned char* buffer, int length);
	void readResult(logkinect::Color_Packet& packet);
	void bufferDecompress();

private:
	bool mStop;
	FarsanThread mThread;
	FarsanLock mRgbMutex;
	FarsanCondVar mRgbLockCv;
	
	//std::thread spawn();
	

};

}

#endif
