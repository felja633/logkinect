#ifndef BUFFER_HANDLER_H
#define BUFFER_HANDLER_H
#include "read_file_handler.h"
#include <string>
#include "farsan_thread.h"

#include <stdio.h>

#ifdef _WIN32
#include <cstring>
#include <ctime>
#include <sstream>
#else
#if __cplusplus > 199711L
#include <cstring>
#include <ctime>
#include <sstream>
#include <chrono>
typedef std::chrono::duration<int, std::ratio<1, 30>> frame_duration;
#else
#include <cstring>
#include <ctime>
#include <sstream>
#endif
#endif

namespace logkinect
{

struct Relative_Pose_To_Save
{
	public:
		Relative_Pose_To_Save() {}

		~Relative_Pose_To_Save()
		{
			/*delete[] (double*)rotation.p;
			delete[] (double*)translation.p;*/
		}

		hvl_t rotation;

		hvl_t translation;
};


struct Camera_Device_To_Save {

	Camera_Device_To_Save()
	{

	}

	~Camera_Device_To_Save()
	{
	    /*
		delete (double*)distortion_coefficients.p;
		delete (double*)intrinsic_camera_matrix.p;
		*/
	}

	hvl_t distortion_coefficients;
	int num_distortion_coefficients;
	hvl_t intrinsic_camera_matrix;

	int camera_type;
};

struct tmp_ir_struct
{
	float fx, fy, cx, cy, k1, k2, k3, p1, p2;
};

class BufferHandler {
public:
	//BufferHandler(std::string filename);
	virtual ~BufferHandler() {}
	virtual void readNextIrFrame(unsigned char** ir_buffer, int* ir_buffer_length) = 0;
	virtual void readNextRgbFrame(unsigned char** rgb_buffer, int* rgb_buffer_length) = 0;
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual bool running() = 0;
	unsigned char* m_p0TableBuffer;
	int m_p0TableLength;
	double* rot, *trans;
	double ir_intrinsics[7];
	double rgb_intrinsics[7];
};

class LogBufferHandler: public BufferHandler {

public:
	LogBufferHandler(std::string filename);
	virtual ~LogBufferHandler();
	virtual void readNextIrFrame(unsigned char** ir_buffer, int* ir_buffer_length);
	virtual void readNextRgbFrame(unsigned char** rgb_buffer, int* rgb_buffer_length);
	virtual void start();
	virtual void stop();
	virtual bool running();
	void bufferUpdateThread();
private:
	void readFrame(unsigned char** ir_buffer,unsigned char** rgb_buffer, int* ir_buffer_length, int* rgb_buffer_length);
	//std::thread spawn();
	void initializeIrCameraFromFile2(double** dist, double** intr, std::string filename);
	void initializeCameraFromFile(double** dist, double** intr, std::string filename, std::string datastname);
	void initializeRelativePoseFromFile(double** rotation, double** translation, std::string filename);
	void readCameraParametersFromFile(double** ir_dist, double** rgb_dist, double** ir_intr, double** rgb_intr, double** rotation, double** translation, std::string filename);

	ReadFileHandler* mFileHandler;
	int mNumberOfFrames;
	int mIrFrameNum, mRgbFrameNum, mReadIrFrameNum, mReadRgbFrameNum;
	unsigned char** mIrBuffer, **mRgbBuffer;
	int* mIrBufferLength, *mRgbBufferLength;
	FarsanThread mThread;
	FarsanLock mIrMutex, mRgbMutex;
	FarsanCondVar mIrLockCv, mRgbLockCv;
	bool mStop;
};
}
#endif
