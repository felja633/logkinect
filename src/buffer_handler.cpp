#include "buffer_handler.h"

static THREAD_RET_VAL el_stupido(THREAD_ARG_TYPE this_is_this)
{
	logkinect::LogBufferHandler* my_this = reinterpret_cast<logkinect::LogBufferHandler*> (this_is_this);
	my_this->bufferUpdateThread();
	return NULL;
}

logkinect::LogBufferHandler::LogBufferHandler(const std::string filename, unsigned int im_num_offset)
{
	mIrBuffer = new unsigned char*[2];
	mRgbBuffer = new unsigned char*[2];
	mIrBufferLength = new int[2];
	mRgbBufferLength = new int[2];

  mFileHandler = new ReadFileHandler(filename);
	mNumberOfFrames = mFileHandler->number_of_groups;

	if(im_num_offset >= mNumberOfFrames)
		im_num_offset = 0;

  mFileHandler->ReadBuffer(&m_p0TableBuffer, &m_p0TableLength, "/P0Tables");
	mIrFrameNum = im_num_offset;
	mRgbFrameNum = im_num_offset;
	mReadIrFrameNum = im_num_offset;
	mReadRgbFrameNum = im_num_offset;

	double* ir_distortion_parameters, *rgb_distortion_parameters, *ir_intrinsic_matrix, *rgb_intrinsic_matrix;
	//readCameraParametersFromFile(&ir_distortion_parameters, &rgb_distortion_parameters, &ir_intrinsic_matrix, &rgb_intrinsic_matrix, &rot, &trans, "/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/kinect_parameters/calib_pose_fixed_ir_2.h5");

	initializeIrCameraFromFile2(&ir_distortion_parameters, &ir_intrinsic_matrix, "/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/kinect_parameters/cam_default_params_1.h5");

	ir_intrinsics[0] = ir_intrinsic_matrix[0];
	ir_intrinsics[1] = ir_intrinsic_matrix[1];
	ir_intrinsics[2] = ir_intrinsic_matrix[2];
	ir_intrinsics[3] = ir_intrinsic_matrix[3];
	ir_intrinsics[4] = ir_distortion_parameters[0];
	ir_intrinsics[5] = ir_distortion_parameters[1];
	ir_intrinsics[6] = ir_distortion_parameters[4];

	rot = new double[3];
	for(unsigned int i = 0; i < 3; i++)
		rot[i] = 0.0;

	trans = new double[3];
	for(unsigned int i = 0; i < 3; i++)
		trans[i] = 0.0;
	/*rgb_intrinsics[0] = rgb_intrinsic_matrix[0];
	rgb_intrinsics[1] = rgb_intrinsic_matrix[1];
	rgb_intrinsics[2] = rgb_intrinsic_matrix[2];
	rgb_intrinsics[3] = rgb_intrinsic_matrix[3];
	rgb_intrinsics[4] = rgb_distortion_parameters[0];
	rgb_intrinsics[5] = rgb_distortion_parameters[1];
	rgb_intrinsics[6] = rgb_distortion_parameters[5];

	delete[] ir_distortion_parameters;
	delete[] rgb_distortion_parameters;
	delete[] ir_intrinsic_matrix;
	delete[] rgb_intrinsic_matrix;*/

}

logkinect::LogBufferHandler::~LogBufferHandler()
{
	std::cout<<"~LogBufferHandler() \n";
	stop();
	//delete[] mIrBuffer;
	//delete[] mRgbBuffer;
  delete mFileHandler;
}

void logkinect::LogBufferHandler::initializeCameraFromFile(double** dist, double** intr, std::string filename, std::string datastname)
{

	hid_t      s1_tid;                          /* File datatype identifier */
  hid_t      file_id, dataset, space, vlen_tid_dist, vlen_tid_intr;  /* Handles */
	Camera_Device_To_Save cam_struct;
  // Open HDF5 file handle, read only
	file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset = H5Dopen(file_id, datastname.c_str(),H5P_DEFAULT);

	space = H5Dget_space(dataset);

	s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(Camera_Device_To_Save));
	vlen_tid_dist = H5Tvlen_create(H5T_NATIVE_DOUBLE);
	vlen_tid_intr = H5Tvlen_create(H5T_NATIVE_DOUBLE);

	H5Tinsert(s1_tid, "distortion_coefficients", HOFFSET(Camera_Device_To_Save, distortion_coefficients), vlen_tid_dist);
	H5Tinsert(s1_tid, "num_distortion_coefficients", HOFFSET(Camera_Device_To_Save, num_distortion_coefficients), H5T_NATIVE_INT);
	H5Tinsert(s1_tid, "intrinsic_camera_matrix", HOFFSET(Camera_Device_To_Save, intrinsic_camera_matrix), vlen_tid_intr);
	H5Tinsert(s1_tid, "camera_type", HOFFSET(Camera_Device_To_Save, camera_type), H5T_NATIVE_INT);

	cam_struct.distortion_coefficients.len = 8;
	cam_struct.distortion_coefficients.p = new double[8];

	for(unsigned int i = 0; i < 8; i++)
		((double*)cam_struct.distortion_coefficients.p)[i] = 0.0;

	cam_struct.intrinsic_camera_matrix.len = 9;
	cam_struct.intrinsic_camera_matrix.p = new double[9];
	for(unsigned int i = 0; i < 9; i++)
		((double*)cam_struct.intrinsic_camera_matrix.p)[i] = 0.0;

	H5Dread(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, &cam_struct);

	*dist = new double[cam_struct.num_distortion_coefficients];
	double* tmp_dist = (double*)cam_struct.distortion_coefficients.p;
	for(int i = 0; i < cam_struct.num_distortion_coefficients; i++)
		(*dist)[i] = tmp_dist[i];


	*intr = new double[4];
	double* tmp_intr = (double*)cam_struct.intrinsic_camera_matrix.p;
  (*intr)[0] = tmp_intr[0];
	(*intr)[1] = tmp_intr[4];
	(*intr)[2] = tmp_intr[2];
	(*intr)[3] = tmp_intr[5];

	H5Tclose(s1_tid);
  H5Sclose(space);
  H5Dclose(dataset);
  H5Fclose(file_id);

}

void logkinect::LogBufferHandler::initializeIrCameraFromFile2(double** dist, double** intr, std::string filename)
{

	hid_t      s1_tid;                          /* File datatype identifier */
  hid_t      file_id, dataset, space;  /* Handles */
  // Open HDF5 file handle, read only
	file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset = H5Dopen(file_id, "ir_parameters",H5P_DEFAULT);

	space = H5Dget_space( dataset ); 
	
	s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(tmp_ir_struct));

	H5Tinsert(s1_tid, "fx", HOFFSET(tmp_ir_struct, fx), H5T_NATIVE_FLOAT);
	H5Tinsert(s1_tid, "fy", HOFFSET(tmp_ir_struct, fy), H5T_NATIVE_FLOAT);
	H5Tinsert(s1_tid, "cx", HOFFSET(tmp_ir_struct, cx), H5T_NATIVE_FLOAT);
	H5Tinsert(s1_tid, "cy", HOFFSET(tmp_ir_struct, cy), H5T_NATIVE_FLOAT);
	H5Tinsert(s1_tid, "k1", HOFFSET(tmp_ir_struct, k1), H5T_NATIVE_FLOAT);
	H5Tinsert(s1_tid, "k2", HOFFSET(tmp_ir_struct, k2), H5T_NATIVE_FLOAT);
	H5Tinsert(s1_tid, "k3", HOFFSET(tmp_ir_struct, k3), H5T_NATIVE_FLOAT);
	H5Tinsert(s1_tid, "p1", HOFFSET(tmp_ir_struct, p1), H5T_NATIVE_FLOAT);	
  H5Tinsert(s1_tid, "p2", HOFFSET(tmp_ir_struct, p2), H5T_NATIVE_FLOAT);	

	tmp_ir_struct tmp_ir;

	H5Dread(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tmp_ir);

	*dist = new double[5];
	(*dist)[0] = (double)tmp_ir.k1;
	(*dist)[1] = (double)tmp_ir.k2;
	(*dist)[2] = (double)tmp_ir.p1;
	(*dist)[3] = (double)tmp_ir.p2;
	(*dist)[4] = (double)tmp_ir.k3;

	*intr = new double[4];
  (*intr)[0] = (double)tmp_ir.fx;
	(*intr)[1] = (double)tmp_ir.fy;
	(*intr)[2] = (double)tmp_ir.cx;
	(*intr)[3] = (double)tmp_ir.cy;

  H5Tclose(s1_tid);
  H5Sclose(space);
  H5Dclose(dataset);
  H5Fclose(file_id);
}

void logkinect::LogBufferHandler::initializeRelativePoseFromFile(double** rotation, double** translation, std::string filename)
{
	Relative_Pose_To_Save pose;
	hid_t      s1_tid;                          /* File datatype identifier */
  hid_t      file_id, dataset, space, vlen_tid_rot, vlen_tid_trans;

	file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	dataset = H5Dopen(file_id, "relative_pose",H5P_DEFAULT);

	space = H5Dget_space(dataset);

	s1_tid = H5Tcreate(H5T_COMPOUND, sizeof(Relative_Pose_To_Save));
	vlen_tid_rot = H5Tvlen_create(H5T_NATIVE_DOUBLE);
	vlen_tid_trans = H5Tvlen_create(H5T_NATIVE_DOUBLE);

	H5Tinsert(s1_tid, "rotation", HOFFSET(Relative_Pose_To_Save, rotation), vlen_tid_rot);
	H5Tinsert(s1_tid, "translation", HOFFSET(Relative_Pose_To_Save, translation), vlen_tid_trans);

	pose.rotation.len = 3;
	pose.rotation.p = new double[3];
	pose.translation.len = 3;
	pose.translation.p = new double[3];

  H5Dread(dataset, s1_tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, &pose);

	*rotation = new double[3];
	double* tmp_rot = (double*)pose.rotation.p;
	for(unsigned int i = 0; i < 3; i++)
		(*rotation)[i] = tmp_rot[i];

	*translation = new double[3];
	double* tmp_trans = (double*)pose.translation.p;
	for(unsigned int i = 0; i < 3; i++)
		(*translation)[i] = tmp_trans[i];

	H5Tclose(s1_tid);
  H5Sclose(space);
  H5Dclose(dataset);
  H5Fclose(file_id);
}

void logkinect::LogBufferHandler::readCameraParametersFromFile(double** ir_dist, double** rgb_dist, double** ir_intr, double** rgb_intr, double** rotation, double** translation, std::string filename)
{
	initializeCameraFromFile(ir_dist, ir_intr, filename, "leader_calibration_parameters");
	initializeCameraFromFile(rgb_dist, rgb_intr, filename, "follower_calibration_parameters");
	initializeRelativePoseFromFile(rotation, translation, filename);
}


void logkinect::LogBufferHandler::readFrame(unsigned char** ir_buffer,unsigned char** rgb_buffer, int* ir_buffer_length, int* rgb_buffer_length)
{
	if(mIrFrameNum != mRgbFrameNum || mIrFrameNum > mNumberOfFrames)
	{
		std::cout<<"ERROR: Frame Num failure \n";
		exit(-1);
		return;
	}

	
  mFileHandler->ReadIrBuffer(ir_buffer, ir_buffer_length, mIrFrameNum);
  mFileHandler->ReadRgbBuffer(rgb_buffer, rgb_buffer_length, mRgbFrameNum);
}

void logkinect::LogBufferHandler::start()
{
	mStop = false;
	//mThread.spawn([this] { this->bufferUpdateThread(); });
	mThread.spawn(&el_stupido, (THREAD_ARG_TYPE)this);
}

void logkinect::LogBufferHandler::stop()
{
	if(mStop)
		return;

	mStop = true;
	mIrLockCv.notifyOne();
	mRgbLockCv.notifyOne();
	std::cout<<"join thread \n";
	mThread.join();
	std::cout<<"thread joined\n";
	//delete[] mIrBuffer[0];
	//delete[] mIrBuffer[1];
	
	//delete[] mRgbBuffer[0];
	//delete[] mRgbBuffer[1];
	
}

bool logkinect::LogBufferHandler::running()
{
#ifdef WIN32
	mIrMutex.lock();
#else
#if __cplusplus > 199711L
	std::unique_lock<std::mutex> ir_lck = mIrMutex.lock();
#else
	mIrMutex.lock();
#endif
#endif
#ifdef WIN32
	mRgbMutex.lock();
#else
#if __cplusplus > 199711L
	std::unique_lock<std::mutex> rgb_lck = mRgbMutex.lock();
#else
	mRgbMutex.lock();
#endif
#endif
	
	bool is_running = (mNumberOfFrames>mRgbFrameNum) && (mNumberOfFrames>mIrFrameNum);

#ifdef WIN32
	//windows
	mRgbMutex.unlock();
#else
#if __cplusplus > 199711L
	mRgbMutex.unlock(rgb_lck);
#else
	mRgbMutex.unlock();
#endif
#endif

#ifdef WIN32
	mIrMutex.unlock();
#else
#if __cplusplus > 199711L
	mIrMutex.unlock(ir_lck);
#else
	mIrMutex.unlock();
#endif
#endif
	return is_running;
}

/*std::thread LogBufferHandler::spawn()
{
	return std::thread( [this] { this->bufferUpdateThread(); } );
}*/

void logkinect::LogBufferHandler::readNextIrFrame(unsigned char** ir_buffer, int* ir_buffer_length)
{
	int index, len;
#ifdef WIN32
	mIrMutex.lock();
#else
#if __cplusplus > 199711L
	std::unique_lock<std::mutex> ir_lck = mIrMutex.lock();
#else
	mIrMutex.lock();
#endif
#endif
	while(!(mIrFrameNum>mReadIrFrameNum))
	{
#ifdef WIN32
    mIrLockCv.wait(&mIrMutex);
#else
#if __cplusplus > 199711L
		mIrLockCv.wait(ir_lck);
#else
		mIrLockCv.wait(&mIrMutex);
#endif
#endif
	}
	//std::cout<<"mReadIrFrameNum = "<<mReadIrFrameNum<<std::endl;
	index = mReadIrFrameNum % 2;
	len = mIrBufferLength[index];
	*ir_buffer = new unsigned char[len];
	std::memcpy (*ir_buffer,mIrBuffer[index],len);
	*ir_buffer_length = len;
	delete[] mIrBuffer[index];
	mReadIrFrameNum++;
	mIrLockCv.notifyOne();
#ifdef WIN32 
	mIrMutex.unlock();
#else
#if __cplusplus > 199711L
	mIrMutex.unlock(ir_lck);
#else
	mIrMutex.unlock();
#endif
#endif
}

void logkinect::LogBufferHandler::readNextRgbFrame(unsigned char** rgb_buffer, int* rgb_buffer_length)
{
	int index, len;
#ifdef WIN32
	mRgbMutex.lock();
#else
#if __cplusplus > 199711L
	std::unique_lock<std::mutex> rgb_lck = mRgbMutex.lock();
#else
	mRgbMutex.lock();
#endif
#endif
	while(!(mRgbFrameNum>mReadRgbFrameNum))
	{
#ifdef  WIN32
	mRgbLockCv.wait(&mRgbMutex);
#else
#if __cplusplus > 199711L
		mRgbLockCv.wait(rgb_lck);
#else
		mRgbLockCv.wait(&mRgbMutex);
#endif
#endif
	}

    //printf("LogBufferHandler - readNextRgbFrame 1\n");
	index = mReadRgbFrameNum % 2;
	len = mRgbBufferLength[index];
	*rgb_buffer = new unsigned char[len];
	std::memcpy (*rgb_buffer,mRgbBuffer[index],len);
	*rgb_buffer_length = len;
	delete[] mRgbBuffer[index];
	mReadRgbFrameNum++;
	mRgbLockCv.notifyOne();
#ifdef  WIN32 
	//windows
	mRgbMutex.unlock();
#else
#if __cplusplus > 199711L
	mRgbMutex.unlock(rgb_lck);
#else
	mRgbMutex.unlock();
#endif
#endif
 //printf("LogBufferHandler - readNextRgbFrame 2\n");
}

void logkinect::LogBufferHandler::bufferUpdateThread()
{
	unsigned char* ir_buffer, *rgb_buffer;
	int ir_buffer_length, rgb_buffer_length, index;
    //printf("LogBufferHandler - bufferUpdateThread 0\n");

	while(!mStop)
	{
#ifdef WIN32
	//windows
#else
#if __cplusplus > 199711L
		auto start_time = std::chrono::steady_clock::now();
		auto end_time  = start_time + frame_duration(1);
#else
#endif
#endif
		if(running())
		{
			//read buffers from file
			readFrame(&ir_buffer, &rgb_buffer, &ir_buffer_length, &rgb_buffer_length);

			//protect buffers
#ifdef WIN32 
			mIrMutex.lock();
#else
#if __cplusplus > 199711L
			std::unique_lock<std::mutex> ir_lck = mIrMutex.lock();
#else
			mIrMutex.lock();
#endif
#endif
			//wait for rendering thread to catch up
			while((mIrFrameNum>mReadIrFrameNum+1) & !mStop)
			{
#ifdef WIN32 
				mIrLockCv.wait(&mIrMutex);
#else
#if __cplusplus > 199711L
				mIrLockCv.wait(ir_lck);
#else
				mIrLockCv.wait(&mIrMutex);
#endif
#endif
			}

			if(mStop)
			{
				delete[] ir_buffer;
				delete[] rgb_buffer;
				std::cout<<"LogBufferHandler thread return \n";
				return;
			}
			//std::cout<<"write ir frame: "<<mIrFrameNum<<std::endl;
			index = mIrFrameNum % 2;

			mIrBuffer[index] = new unsigned char[ir_buffer_length];
			mIrBufferLength[index] = ir_buffer_length;
			std::memcpy (mIrBuffer[index],ir_buffer,ir_buffer_length);
			mIrFrameNum++;
			mIrLockCv.notifyOne();
#ifdef WIN32
			mIrMutex.unlock();
#else
#if __cplusplus > 199711L
			mIrMutex.unlock(ir_lck);
#else
			mIrMutex.unlock();
#endif
#endif

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
		   //   printf("LogBufferHandler - bufferUpdateThread 3\n");
			//wait f|| rendering thread to catch up
			while((mRgbFrameNum>mReadRgbFrameNum+1) & !mStop)
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
				delete[] ir_buffer;
				delete[] rgb_buffer;
				std::cout<<"LogBufferHandler thread return \n";
				return;
			}

			index = mRgbFrameNum % 2;
			mRgbBuffer[index] = new unsigned char[rgb_buffer_length];
			mRgbBufferLength[index] = rgb_buffer_length;
			std::memcpy (mRgbBuffer[index],rgb_buffer,rgb_buffer_length);

			mRgbFrameNum++;
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

			delete[] ir_buffer;
			delete[] rgb_buffer;
		}
#ifdef  WIN32
		//Sleep(2);
		usleep(20000);
#else
#if __cplusplus > 199711L
		std::this_thread::sleep_until(end_time);
#else
		usleep(30000);
#endif
#endif

	}

	std::cout<<"LogBufferHandler thread return \n";
}

