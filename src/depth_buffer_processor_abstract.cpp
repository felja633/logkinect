
#include "depth_buffer_processor.h"
//#include <libfreenect2/protocol/response.h>

//#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>


#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include "resources.inc"

#define M_PI_ 3.14159265359

struct ResourceDescriptor
{
  const char *filename;
  const unsigned char *data;
  size_t length;
};
ResourceDescriptor resource_descriptors[] = {
  { "kinect_parameters/11to16.bin", resource0, sizeof(resource0) },
  { "kinect_parameters/xTable.bin", resource1, sizeof(resource1) },
  { "kinect_parameters/zTable.bin", resource2, sizeof(resource2) },
  { "opencl_depth_packet_processor.cl", resource9, sizeof(resource9) },
};
int resource_descriptors_length = 4;

bool loadResource(const std::string &name, unsigned char const**data, size_t *length)
{
  bool result = false;

  for(int i = 0; i < resource_descriptors_length; ++i)
  {
    if(name.compare(resource_descriptors[i].filename) == 0)
    {
      *data = resource_descriptors[i].data;
      *length = resource_descriptors[i].length;
      result = true;
      break;
    }
  }
  return result;
}

bool loadBufferFromResources(const std::string &filename, unsigned char *buffer, const size_t n)
{
  size_t length = 0;
  const unsigned char *data = NULL;

  if(!loadResource(filename, &data, &length))
  {
    std::cerr << "failed to load resource: " << filename << std::endl;
    return false;
  }

  if(length != n)
  {
    std::cerr << "wrong size of resource: " << filename << std::endl;
    return false;
  }

  memcpy(buffer, data, length);
  return true;
}


std::string loadCLSource(const std::string &filename)
{
  //const unsigned char *data;
  //size_t length = 0;

  std::ifstream ifs(filename.c_str());
  std::string content( (std::istreambuf_iterator<char>(ifs) ),
                       (std::istreambuf_iterator<char>()    ) );
/*
  if(!loadResource(filename, &data, &length))
  {
    std::cerr << "failed to load cl source!" << std::endl;
    return "";
  }
*/
  //return std::string(reinterpret_cast<const char *>(data), length);
  ifs.close();
  return content;
}

static void saveFloatArrayToFileBin(float* pts, int len, std::string name)
{
	std::ofstream fout(name.c_str(), std::ofstream::binary);

 	if(fout.is_open())
	{
		//for(int i = 0; i < len; i++)
		//	fout<<pts[i];
		fout.seekp(0,std::ofstream::end);
		fout.write((char*)pts,len*sizeof(float));
		fout.close();
	}
	else
	{
		std::cout<<"unable to open file\n";
	}
}

logkinect::OpenCLDepthBufferProcessorImpl::~OpenCLDepthBufferProcessorImpl()
{
	delete lut11to16;
  delete x_table;
  delete z_table;
  delete p0_table;
}

void logkinect::OpenCLDepthBufferProcessorImpl::generateOptions(std::string &options) const
{
    std::ostringstream oss;
    oss.precision(16);
    oss << std::scientific;
    oss << " -D BFI_BITMASK=" << "0x180";

    oss << " -D AB_MULTIPLIER=" << params.ab_multiplier << "f";
    oss << " -D AB_MULTIPLIER_PER_FRQ0=" << params.ab_multiplier_per_frq[0] << "f";
    oss << " -D AB_MULTIPLIER_PER_FRQ1=" << params.ab_multiplier_per_frq[1] << "f";
    oss << " -D AB_MULTIPLIER_PER_FRQ2=" << params.ab_multiplier_per_frq[2] << "f";
    oss << " -D AB_OUTPUT_MULTIPLIER=" << params.ab_output_multiplier << "f";

    oss << " -D PHASE_IN_RAD0=" << params.phase_in_rad[0] << "f";
    oss << " -D PHASE_IN_RAD1=" << params.phase_in_rad[1] << "f";
    oss << " -D PHASE_IN_RAD2=" << params.phase_in_rad[2] << "f";

    oss << " -D JOINT_BILATERAL_AB_THRESHOLD=" << params.joint_bilateral_ab_threshold << "f";
    oss << " -D JOINT_BILATERAL_MAX_EDGE=" << params.joint_bilateral_max_edge << "f";
    oss << " -D JOINT_BILATERAL_EXP=" << params.joint_bilateral_exp << "f";
    oss << " -D JOINT_BILATERAL_THRESHOLD=" << (params.joint_bilateral_ab_threshold * params.joint_bilateral_ab_threshold) / (params.ab_multiplier * params.ab_multiplier) << "f";
    oss << " -D GAUSSIAN_KERNEL_0=" << params.gaussian_kernel[0] << "f";
    oss << " -D GAUSSIAN_KERNEL_1=" << params.gaussian_kernel[1] << "f";
    oss << " -D GAUSSIAN_KERNEL_2=" << params.gaussian_kernel[2] << "f";
    oss << " -D GAUSSIAN_KERNEL_3=" << params.gaussian_kernel[3] << "f";
    oss << " -D GAUSSIAN_KERNEL_4=" << params.gaussian_kernel[4] << "f";
    oss << " -D GAUSSIAN_KERNEL_5=" << params.gaussian_kernel[5] << "f";
    oss << " -D GAUSSIAN_KERNEL_6=" << params.gaussian_kernel[6] << "f";
    oss << " -D GAUSSIAN_KERNEL_7=" << params.gaussian_kernel[7] << "f";
    oss << " -D GAUSSIAN_KERNEL_8=" << params.gaussian_kernel[8] << "f";

    oss << " -D PHASE_OFFSET=" << params.phase_offset << "f";
    oss << " -D UNAMBIGIOUS_DIST=" << params.unambigious_dist << "f";
    oss << " -D INDIVIDUAL_AB_THRESHOLD=" << params.individual_ab_threshold << "f";
    oss << " -D AB_THRESHOLD=" << params.ab_threshold << "f";
    oss << " -D AB_CONFIDENCE_SLOPE=" << params.ab_confidence_slope << "f";
    oss << " -D AB_CONFIDENCE_OFFSET=" << params.ab_confidence_offset << "f";
    oss << " -D MIN_DEALIAS_CONFIDENCE=" << params.min_dealias_confidence << "f";
    oss << " -D MAX_DEALIAS_CONFIDENCE=" << params.max_dealias_confidence << "f";

    oss << " -D EDGE_AB_AVG_MIN_VALUE=" << params.edge_ab_avg_min_value << "f";
    oss << " -D EDGE_AB_STD_DEV_THRESHOLD=" << params.edge_ab_std_dev_threshold << "f";
    oss << " -D EDGE_CLOSE_DELTA_THRESHOLD=" << params.edge_close_delta_threshold << "f";
    oss << " -D EDGE_FAR_DELTA_THRESHOLD=" << params.edge_far_delta_threshold << "f";
    oss << " -D EDGE_MAX_DELTA_THRESHOLD=" << params.edge_max_delta_threshold << "f";
    oss << " -D EDGE_AVG_DELTA_THRESHOLD=" << params.edge_avg_delta_threshold << "f";
    oss << " -D MAX_EDGE_COUNT=" << params.max_edge_count << "f";

    oss << " -D MIN_DEPTH=" << params.min_depth * 1000.0f << "f";
    oss << " -D MAX_DEPTH=" << params.max_depth * 1000.0f << "f";

		oss << " -D NUM_CHANNELS="<<params.num_channels;
		oss << " -D CHANNEL_FILT_SIZE="<<params.channel_filt_size;
		oss << " -D CHANNEL_CONFIDENCE_SCALE="<<params.channel_confidence_scale<<"f";
		oss << " -D BLOCK_SIZE_COL="<<params.block_size_col;
		oss << " -D BLOCK_SIZE_ROW="<<params.block_size_row;

    options = oss.str();
}

void logkinect::OpenCLDepthBufferProcessorImpl::getDevices(const std::vector<cl::Platform> &platforms, std::vector<cl::Device> &devices)
{
  devices.clear();
  for(size_t i = 0; i < platforms.size(); ++i)
  {
    const cl::Platform &platform = platforms[i];

    std::vector<cl::Device> devs;
    if(platform.getDevices(CL_DEVICE_TYPE_ALL, &devs) != CL_SUCCESS)
    {
      continue;
    }

    devices.insert(devices.end(), devs.begin(), devs.end());
  }
}

void logkinect::OpenCLDepthBufferProcessorImpl::listDevice(std::vector<cl::Device> &devices)
{
  std::cout << "listDevice devices:" << std::endl;
  for(size_t i = 0; i < devices.size(); ++i)
  {
    cl::Device &dev = devices[i];
    std::string devName, devVendor, devType;
    size_t devTypeID;
    dev.getInfo(CL_DEVICE_NAME, &devName);
    dev.getInfo(CL_DEVICE_VENDOR, &devVendor);
    dev.getInfo(CL_DEVICE_TYPE, &devTypeID);

    switch(devTypeID)
    {
    case CL_DEVICE_TYPE_CPU:
      devType = "CPU";
      break;
    case CL_DEVICE_TYPE_GPU:
      devType = "GPU";
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      devType = "ACCELERATOR";
      break;
    case CL_DEVICE_TYPE_CUSTOM:
      devType = "CUSTOM";
      break;
    default:
      devType = "UNKNOWN";
    }

    std::cout << "  " << i << ": " << devName << " (" << devType << ")[" << devVendor << ']' << std::endl;
  }
}

bool logkinect::OpenCLDepthBufferProcessorImpl::selectDevice(std::vector<cl::Device> &devices, const int deviceId)
{
  if(deviceId != -1 && devices.size() > (size_t)deviceId)
  {
    device = devices[deviceId];
    return true;
  }

  bool selected = false;
  size_t selectedType = 0;

  for(size_t i = 0; i < devices.size(); ++i)
  {
    cl::Device &dev = devices[i];
    size_t devTypeID;
    dev.getInfo(CL_DEVICE_TYPE, &devTypeID);

    if(!selected || (selectedType != CL_DEVICE_TYPE_GPU && devTypeID == CL_DEVICE_TYPE_GPU))
    {
      selectedType = devTypeID;
      selected = true;
      device = dev;
    }
  }
  return selected;
}

bool logkinect::OpenCLDepthBufferProcessorImpl::initDevice(const int deviceId)
{
  if(!readProgram(sourceCode))
  {
		std::cout<<"logkinect::OpenCLDepthBufferProcessorImpl::initDevice: unable to read program \n";
    return false;
  }

  cl_int err = CL_SUCCESS;

  std::vector<cl::Platform> platforms;
  if(cl::Platform::get(&platforms) != CL_SUCCESS)
  {
    std::cerr << "init error while getting opencl platforms." << std::endl;
    return false;
  }
  if(platforms.empty())
  {
    std::cerr << "init no opencl platforms found." << std::endl;
    return false;
  }

  std::vector<cl::Device> devices;
  getDevices(platforms, devices);
  listDevice(devices);
  if(selectDevice(devices, deviceId))
  {
    std::string devName, devVendor, devType;
    size_t devTypeID;
    device.getInfo(CL_DEVICE_NAME, &devName);
    device.getInfo(CL_DEVICE_VENDOR, &devVendor);
    device.getInfo(CL_DEVICE_TYPE, &devTypeID);

    switch(devTypeID)
    {
    case CL_DEVICE_TYPE_CPU:
      devType = "CPU";
      break;
    case CL_DEVICE_TYPE_GPU:
      devType = "GPU";
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      devType = "ACCELERATOR";
      break;
    case CL_DEVICE_TYPE_CUSTOM:
      devType = "CUSTOM";
      break;
    default:
      devType = "UNKNOWN";
    }
    std::cout << "init selected device: " << devName << " (" << devType << ")[" << devVendor << ']' << std::endl;
  }
  else
  {
    std::cerr << "init" "could not find any suitable device" << std::endl;
    return false;
  }

  mContext = cl::Context(device);
	std::cout<<"logkinect::OpenCLDepthBufferProcessorImpl::initDevice: Device initialized \n";
  return true;
}

void logkinect::OpenCLDepthBufferProcessorImpl::fill_trig_table(const P0TablesResponse *p0table)
{
/*
    for(unsigned int i = 0; i<424*512;i++)
    {
      uint16_t allan16 = p0table->p0table0[i];
      uint32_t allan32 = 0;
      allan32 = allan16;
      float allanf   = *(float *)&allan32;
      float tmp = -&((float* )&it0) * 0.000031 * M_PI_;
      p0_table[i].s[1] = -((float)(p0table->p0table1[i]))* 0.000031 * M_PI_;
      p0_table[i].s[2] = -((float)(p0table->p0table2[i]))* 0.000031 * M_PI_;
      p0_table[i].s[3] = 0.0f;
						it->s[0] = -(*((float*) it0)) * 0.000031 * M_PI_; //((float) * it0);//-((float) * it0) * 0.000031 * M_PI_;
			it->s[1] = -(*((float*) it1)) * 0.000031 * M_PI_;//-((float) * it1) * 0.000031 * M_PI_;
			it->s[2] = -(*((float*) it2)) * 0.000031 * M_PI_;//-((float) * it2) * 0.000031 * M_PI_;
			it->s[3] = 0.0f;
    }
*/

	for(int r = 0; r < 424; ++r)
	{
		cl_float3 *it = &p0_table[r * 512];
		const uint16_t *it0 = &p0table->p0table0[r * 512];
		const uint16_t *it1 = &p0table->p0table1[r * 512];
		const uint16_t *it2 = &p0table->p0table2[r * 512];
		//std::cout<<r<<std::endl;
		for(int c = 0; c < 512; ++c, ++it, ++it0, ++it1, ++it2)
		{
			it->s[0] = -((float) * it0) * 0.000031 * M_PI_;
			it->s[1] = -((float) * it1) * 0.000031 * M_PI_;
			it->s[2] = -((float) * it2) * 0.000031 * M_PI_;
			it->s[3] = 0.0f;
		}
	}

}

void logkinect::OpenCLDepthBufferProcessorImpl::initNewPacket()
{
	if(pipeline_ == 4)
	{
		std::cout<<"read data\n"; 
		packet_.buffer = new unsigned char[packet_.width*packet_.height*packet_.bytes_per_element*4];
		if(dev_mode_>0)
			ir_packet_.buffer = new unsigned char[ir_packet_.width*ir_packet_.height*ir_packet_.bytes_per_element*6];
	}
	else
		packet_.buffer = new unsigned char[packet_.width*packet_.height*packet_.bytes_per_element*2];

}

void logkinect::OpenCLDepthBufferProcessorImpl::initInitProgram()
{
	if(!deviceInitialized)
  {
		std::cout<<"logkinect::OpenCLDepthBufferProcessorImpl::initProgram: Device not initialized \n";
    return false;
  }

  cl_int err = CL_SUCCESS;

  std::string options;
  generateOptions(options);

  cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
  program = cl::Program(mContext, source);
	std::cout<<"build program.. "<<std::endl;
  program.build(options.c_str());
	queue = cl::CommandQueue(mContext, device, 0, &err);
	std::cout<<"create cl::CommandQueue "<<std::endl;
  std::string str;
  program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &str);
  std::cout<<"build log: "<<str<<std::endl;
  //Read only
  buf_lut11to16_size = 2048 * sizeof(cl_short);
  buf_p0_table_size = image_size * sizeof(cl_float3);
  buf_x_table_size = image_size * sizeof(cl_float);
  buf_z_table_size = image_size * sizeof(cl_float);
  buf_buffer_size = ((image_size * 11) / 16) * 10 * sizeof(cl_ushort);
	buf_ir_camera_intrinsics_size = sizeof(cl_float)*7;
	buf_rel_rot_size = sizeof(cl_float)*9;
	buf_rel_trans_size = sizeof(cl_float)*3;

  buf_lut11to16 = cl::Buffer(mContext, CL_READ_ONLY_CACHE, buf_lut11to16_size, NULL, &err);
  buf_p0_table = cl::Buffer(mContext, CL_READ_ONLY_CACHE, buf_p0_table_size, NULL, &err);
  buf_x_table = cl::Buffer(mContext, CL_READ_ONLY_CACHE, buf_x_table_size, NULL, &err);
  buf_z_table = cl::Buffer(mContext, CL_READ_ONLY_CACHE, buf_z_table_size, NULL, &err);
  buf_buffer = cl::Buffer(mContext, CL_READ_ONLY_CACHE, buf_buffer_size, NULL, &err);

 	//Read-Write
  buf_a_size = image_size * sizeof(cl_float3);
  buf_b_size = image_size * sizeof(cl_float3);
  buf_n_size = image_size * sizeof(cl_float3);

  buf_a_filtered_size = image_size * sizeof(cl_float3);
  buf_b_filtered_size = image_size * sizeof(cl_float3);
  buf_edge_test_size = image_size * sizeof(cl_uchar);

	buf_ir_size = image_size * sizeof(cl_float);
	buf_filtered_size = 2*image_size * sizeof(cl_float);

	std::cout<<"initializing OpenCL kernels"<<err<<"\n";
  kernel_processPixelStage1 = cl::Kernel(program, "processPixelStage1", &err);
  kernel_processPixelStage1.setArg(0, buf_lut11to16);
  kernel_processPixelStage1.setArg(1, buf_z_table);
  kernel_processPixelStage1.setArg(2, buf_p0_table);
  kernel_processPixelStage1.setArg(3, buf_buffer);
  kernel_processPixelStage1.setArg(4, buf_a);
  kernel_processPixelStage1.setArg(5, buf_b);
  kernel_processPixelStage1.setArg(6, buf_n);
	kernel_processPixelStage1.setArg(7, buf_ir);

	std::cout<<"processPixelStage1 kernel "<<err<<"\n";

  kernel_filterPixelStage1 = cl::Kernel(program, "filterPixelStage1", &err);
  kernel_filterPixelStage1.setArg(0, buf_a);
  kernel_filterPixelStage1.setArg(1, buf_b);
  kernel_filterPixelStage1.setArg(2, buf_n);
  kernel_filterPixelStage1.setArg(3, buf_a_filtered);
  kernel_filterPixelStage1.setArg(4, buf_b_filtered);
  kernel_filterPixelStage1.setArg(5, buf_edge_test);
}



bool logkinect::OpenCLLibfreenectFullmaskDepthBufferProcessorImpl::initProgram()
{
	initInitProgram();
	std::cout<<"initializing OpenCL buffers"<<std::endl;
	std::cout<<"bilateral filter = "<<bilateral_filter_<<std::endl;
	if(bilateral_filter_ != 1)
		std::cout<<"no bilateral filter is used\n";

	buf_ir_sum_size = image_size * sizeof(cl_float);
	buf_depth_size = 2*image_size * sizeof(cl_float);

	kernel_processPixelStage2_fullmask = cl::Kernel(program, "processPixelStage2_fullmask", &err);
	kernel_processPixelStage2_fullmask.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
	kernel_processPixelStage2_fullmask.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
	kernel_processPixelStage2_fullmask.setArg(2, buf_x_table);
	kernel_processPixelStage2_fullmask.setArg(3, buf_z_table);
	kernel_processPixelStage2_fullmask.setArg(4, buf_depth);
	kernel_processPixelStage2_fullmask.setArg(5, buf_ir_sum);

  kernel_filterPixelStage2 = cl::Kernel(program, "filterPixelStage2", &err);
	kernel_filterPixelStage2.setArg(0, buf_depth);
	kernel_filterPixelStage2.setArg(1, buf_ir_sum);
	kernel_filterPixelStage2.setArg(2, buf_edge_test);
	kernel_filterPixelStage2.setArg(3, buf_filtered);

	if(undistort_ == 1)
	{
		kernel_undistort = cl::Kernel(program, "undistort", &err);
		kernel_undistort.setArg(0, buf_filtered);
		kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
		kernel_undistort.setArg(2, buf_depth_undistorted);
	}
	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4;
  queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
  queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
  queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
  queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
	queue.enqueueWriteBuffer(buf_ir_camera_intrinsics, CL_FALSE, 0, buf_ir_camera_intrinsics_size, ir_camera_intrinsics, NULL, &event4);

  event0.wait();
  event1.wait();
  event2.wait();
  event3.wait();
	event4.wait();

  programInitialized = true;
	std::cout<<"OpenCL program initialized\n";
  return true;
}

void logkinect::OpenCLLibfreenectDepthBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
{
	
	static int cnt = 0;
	unsigned int offset = 0;

  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	if(bilateral_filter_ == 1)
  	queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	queue.enqueueNDRangeKernel(kernel_processPixelStage2_fullmask, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	if(undistort_ == 1)
	{
		queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
		queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
	}
	else
		queue.enqueueReadBuffer(buf_filtered, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);

	queue.finish();
}

bool logkinect::OpenCLLibfreenectDepthBufferProcessorImpl::readProgram(std::string &source) const
{
	source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_depth_packet_processor.cl");
	std::cout<<"load source: /home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_depth_packet_processor.cl \n";
	break;
}

bool logkinect::OpenCLLibfreenectNomaskDepthBufferProcessorImpl::initProgram()
{
	initInitProgram();
	std::cout<<"initializing OpenCL buffers"<<std::endl;
	std::cout<<"bilateral filter = "<<bilateral_filter_<<std::endl;
	if(bilateral_filter_ != 1)
		std::cout<<"no bilateral filter is used\n";


	buf_depth_size = 2*image_size * sizeof(cl_float);
	buf_ir_sum_size = image_size * sizeof(cl_float);

	kernel_processPixelStage2_nomask = cl::Kernel(program, "processPixelStage2_nomask", &err);
	kernel_processPixelStage2_nomask.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
	kernel_processPixelStage2_nomask.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
	kernel_processPixelStage2_nomask.setArg(2, buf_x_table);
	kernel_processPixelStage2_nomask.setArg(3, buf_z_table);
	kernel_processPixelStage2_nomask.setArg(4, buf_depth);

	if(undistort_ == 1)
	{
		kernel_undistort = cl::Kernel(program, "undistort", &err);
		kernel_undistort.setArg(0, buf_depth);
		kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
		kernel_undistort.setArg(2, buf_depth_undistorted);
	}
	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4;
  queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
  queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
  queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
  queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
	queue.enqueueWriteBuffer(buf_ir_camera_intrinsics, CL_FALSE, 0, buf_ir_camera_intrinsics_size, ir_camera_intrinsics, NULL, &event4);

  event0.wait();
  event1.wait();
  event2.wait();
  event3.wait();
	event4.wait();

  programInitialized = true;
	std::cout<<"OpenCL program initialized\n";
  return true;
}

void logkinect::OpenCLLibfreenectNomaskDepthBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
{
	
	static int cnt = 0;
	unsigned int offset = 0;

  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	if(bilateral_filter_ == 1)
  	queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	queue.enqueueNDRangeKernel(kernel_processPixelStage2_nomask, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	if(undistort_ == 1)
	{
		queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
		queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
	}
	else
		queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);


	queue.finish();
}

bool logkinect::OpenCLLibfreenectNomaskDepthBufferProcessorImpl::readProgram(std::string &source) const
{
	source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_depth_packet_processor.cl");
	std::cout<<"load source: /home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_depth_packet_processor.cl \n";

}

bool logkinect::OpenCLHypDepthBufferProcessorImpl::initProgram()
{
	initInitProgram();
	std::cout<<"initializing OpenCL buffers"<<std::endl;
	std::cout<<"bilateral filter = "<<bilateral_filter_<<std::endl;
	if(bilateral_filter_ != 1)
		std::cout<<"no bilateral filter is used\n";


	buf_depth_size = image_size * sizeof(cl_float)*4;
	buf_ir_sum_size = image_size * sizeof(cl_float)*3;

	kernel_processPixelStage2_phase_depth = cl::Kernel(program, "processPixelStage2_phase_depth", &err);
	kernel_processPixelStage2_phase_depth.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
	kernel_processPixelStage2_phase_depth.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
	kernel_processPixelStage2_phase_depth.setArg(2, buf_x_table);
	kernel_processPixelStage2_phase_depth.setArg(3, buf_z_table);
	kernel_processPixelStage2_phase_depth.setArg(4, buf_depth);
	kernel_processPixelStage2_phase_depth.setArg(5, buf_ir_sum);

	if(undistort_ == 1)
	{
		kernel_undistort = cl::Kernel(program, "undistort", &err);
		kernel_undistort.setArg(0, buf_depth);
		kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
		kernel_undistort.setArg(2, buf_depth_undistorted);
	}
	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4;
  queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
  queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
  queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
  queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
	queue.enqueueWriteBuffer(buf_ir_camera_intrinsics, CL_FALSE, 0, buf_ir_camera_intrinsics_size, ir_camera_intrinsics, NULL, &event4);

  event0.wait();
  event1.wait();
  event2.wait();
  event3.wait();
	event4.wait();

  programInitialized = true;
	std::cout<<"OpenCL program initialized\n";
  return true;
}

void logkinect::OpenCLHypDepthBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
{
	
	static int cnt = 0;
	unsigned int offset = 0;

  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	if(bilateral_filter_ == 1)
  	queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase_depth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
	if(dev_mode_>0)
	{
		queue.enqueueReadBuffer(buf_ir_sum, CL_FALSE, 0, buf_ir_sum_size, ir_packet_.buffer, NULL, NULL);
	}

	queue.finish();
}

bool logkinect::OpenCLHypDepthBufferProcessorImpl::readProgram(std::string &source) const
{
	source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl");
	std::cout<<"load source: /home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl \n";

}

bool logkinect::OpenCLSpatialPropagationDepthBufferProcessorImpl::initProgram()
{
	initInitProgram();
	std::cout<<"initializing OpenCL buffers"<<std::endl;
	std::cout<<"bilateral filter = "<<bilateral_filter_<<std::endl;
	if(bilateral_filter_ != 1)
		std::cout<<"no bilateral filter is used\n";


	buf_depth_size = image_size * sizeof(cl_float)*4;
	buf_ir_sum_size = image_size * sizeof(cl_float)*3;
	buf_phase_size = image_size * sizeof(cl_float);

	buf_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	buf_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	buf_phase_3 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);

	buf_w1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	buf_w2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	buf_w3 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	buf_conf1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_conf2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_count = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_count_size, NULL, &err);
	buf_phase_prop_vertical = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_phase_prop_horizontal = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_cost_prop_vertical =  cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_cost_prop_horizontal =  cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);

	kernel_processPixelStage2_phase = cl::Kernel(program, "processPixelStage2_phase", &err);
	kernel_processPixelStage2_phase.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
	kernel_processPixelStage2_phase.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
	kernel_processPixelStage2_phase.setArg(2, buf_phase_1);
	kernel_processPixelStage2_phase.setArg(3, buf_phase_2);
	kernel_processPixelStage2_phase.setArg(4, buf_phase_prop_vertical);
	kernel_processPixelStage2_phase.setArg(5, buf_phase_prop_horizontal);
	kernel_processPixelStage2_phase.setArg(6, buf_w1);
	kernel_processPixelStage2_phase.setArg(7, buf_w2);
	kernel_processPixelStage2_phase.setArg(8, buf_cost_prop_vertical);
	kernel_processPixelStage2_phase.setArg(9, buf_cost_prop_horizontal);
	kernel_processPixelStage2_phase.setArg(10, buf_conf1);
	kernel_processPixelStage2_phase.setArg(11, buf_conf2);
	kernel_processPixelStage2_phase.setArg(12, buf_ir_sum);

	kernel_propagate_vertical = cl::Kernel(program, "propagateVertical", &err);
	kernel_propagate_vertical.setArg(0, buf_phase_1);
	kernel_propagate_vertical.setArg(1, buf_phase_2);
	kernel_propagate_vertical.setArg(2, buf_w1);
	kernel_propagate_vertical.setArg(3, buf_w2);
	kernel_propagate_vertical.setArg(4, buf_conf1);
	kernel_propagate_vertical.setArg(5, buf_conf2);
	kernel_propagate_vertical.setArg(6, buf_count);
	kernel_propagate_vertical.setArg(7, buf_phase_prop_vertical);
	kernel_propagate_vertical.setArg(8, buf_cost_prop_vertical);

	kernel_propagate_horizontal = cl::Kernel(program, "propagateHorizontal", &err);
	kernel_propagate_horizontal.setArg(0, buf_phase_1);
	kernel_propagate_horizontal.setArg(1, buf_phase_2);
	kernel_propagate_horizontal.setArg(2, buf_w1);
	kernel_propagate_horizontal.setArg(3, buf_w2);
	kernel_propagate_horizontal.setArg(4, buf_conf1);
	kernel_propagate_horizontal.setArg(5, buf_conf2);
	kernel_propagate_horizontal.setArg(6, buf_count);
	kernel_propagate_horizontal.setArg(7, buf_phase_prop_horizontal);
	kernel_propagate_horizontal.setArg(8, buf_cost_prop_horizontal);

	kernel_processPixelStage2_depth = cl::Kernel(program, "processPixelStage2_depth", &err);
	//kernel_processPixelStage2_depth.setArg(0, buf_phase);
	kernel_processPixelStage2_depth.setArg(0, buf_phase_prop_vertical);
	kernel_processPixelStage2_depth.setArg(1, buf_phase_prop_horizontal);
	kernel_processPixelStage2_depth.setArg(2, buf_cost_prop_vertical);
	kernel_processPixelStage2_depth.setArg(3, buf_cost_prop_horizontal);
	kernel_processPixelStage2_depth.setArg(4, buf_x_table);
	kernel_processPixelStage2_depth.setArg(5, buf_z_table);
	kernel_processPixelStage2_depth.setArg(6, buf_depth);
	std::cout<<"propagateHorizontal kernel \n";

	if(undistort_ == 1)
	{
		kernel_undistort = cl::Kernel(program, "undistort", &err);
		kernel_undistort.setArg(0, buf_depth);
		kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
		kernel_undistort.setArg(2, buf_depth_undistorted);
	}
	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4;
  queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
  queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
  queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
  queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
	queue.enqueueWriteBuffer(buf_ir_camera_intrinsics, CL_FALSE, 0, buf_ir_camera_intrinsics_size, ir_camera_intrinsics, NULL, &event4);

  event0.wait();
  event1.wait();
  event2.wait();
  event3.wait();
	event4.wait();

  programInitialized = true;
	std::cout<<"OpenCL program initialized\n";
  return true;
}

void logkinect::OpenCLSpatialPropagationDepthBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
{
	
	static int cnt = 0;
	unsigned int offset = 0;

  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	if(bilateral_filter_ == 1)
  	queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	for(unsigned int count = offset; count < 256; count++)
	{
		if(count == offset)
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, NULL, NULL);
		}
		else
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, NULL, NULL);
		}
		queue.enqueueNDRangeKernel(kernel_propagate_vertical, cl::NullRange, 2*424, cl::NullRange, NULL, NULL);
	}
	for(unsigned int count = offset; count < 212; count++)
	{
		if(count == offset)
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, NULL, NULL);
		}
		else
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, NULL, NULL);
		}
		queue.enqueueNDRangeKernel(kernel_propagate_horizontal, cl::NullRange, 2*512, cl::NullRange, NULL, NULL);
	}

	queue.enqueueNDRangeKernel(kernel_processPixelStage2_depth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	if(undistort_ == 1)
	{
		queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
		queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
	}
	else
		queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);

	queue.finish();
}

bool logkinect::OpenCLSpatialPropagationDepthBufferProcessorImpl::readProgram(std::string &source) const
{
	source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl");
	std::cout<<"load source: /home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl \n";
}

bool logkinect::OpenCLChannelsBufferProcessorImpl::initProgram()
{
	initInitProgram();
	std::cout<<"initializing OpenCL buffers"<<std::endl;
	std::cout<<"bilateral filter = "<<bilateral_filter_<<std::endl;
	if(bilateral_filter_ != 1)
		std::cout<<"no bilateral filter is used\n";


	buf_depth_size = image_size * sizeof(cl_float)*4;
	buf_ir_sum_size = image_size * sizeof(cl_float)*3;
	buf_gaussian_kernel_size = (2*params.channel_filt_size+1)*sizeof(cl_float);
	buf_gaussian_kernel = cl::Buffer(mContext, CL_READ_ONLY_CACHE, buf_gaussian_kernel_size, NULL, &err);
	buf_channels_size = image_size * sizeof(cl_float16);

	buf_channels_1_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	buf_channels_2_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);

	
	buf_channels_1_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	buf_channels_2_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	buf_channels_1_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	buf_channels_2_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	buf_channels_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	buf_channels_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);

	if(pipeline_ == 7)
	{
		std::cout<<"allocate extra channel buffers\n";
		buf_channels_3 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_4 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_3_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_4_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_3_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_4_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	}

	switch(pipeline_)
	{
		case 5:
			kernel_processPixelStage1.setArg(8, buf_o);
			kernel_filterPixelStage1.setArg(6, buf_o);
			kernel_filterPixelStage1.setArg(7, buf_o_filtered);

			kernel_processPixelStage2_phase_channels = cl::Kernel(program, "processPixelStage2_phase_channels", &err);
			kernel_processPixelStage2_phase_channels.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_phase_channels.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_phase_channels.setArg(2, buf_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(3, buf_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(4, buf_ir_sum);
			kernel_processPixelStage2_phase_channels.setArg(5, buf_channels_1);
			kernel_processPixelStage2_phase_channels.setArg(6, buf_channels_2);
			kernel_processPixelStage2_phase_channels.setArg(7, buf_channels_1_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(8, buf_channels_2_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(9, buf_channels_1_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(10, buf_channels_2_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(11, buf_ir_camera_intrinsics);
			kernel_processPixelStage2_phase_channels.setArg(12, buf_o_filtered);

			kernel_filter_channels = cl::Kernel(program, "filter_channels", &err);
			kernel_filter_channels.setArg(0, buf_channels_1);
			kernel_filter_channels.setArg(1, buf_channels_2);
			kernel_filter_channels.setArg(2, buf_channels_1_filtered);
			kernel_filter_channels.setArg(3, buf_channels_2_filtered);
			kernel_filter_channels.setArg(4, buf_gaussian_kernel);

			kernel_processPixelStage2_depth_channels = cl::Kernel(program, "processPixelStage2_depth_channels", &err);
			kernel_processPixelStage2_depth_channels.setArg(0, buf_phase_1);
			kernel_processPixelStage2_depth_channels.setArg(1, buf_phase_2);
			kernel_processPixelStage2_depth_channels.setArg(2, buf_channels_1_filtered);
			kernel_processPixelStage2_depth_channels.setArg(3, buf_channels_2_filtered);
			kernel_processPixelStage2_depth_channels.setArg(4, buf_channels_1_phase_1);
			kernel_processPixelStage2_depth_channels.setArg(5, buf_channels_2_phase_1);
			kernel_processPixelStage2_depth_channels.setArg(6, buf_channels_1_phase_2);
			kernel_processPixelStage2_depth_channels.setArg(7, buf_channels_2_phase_2);
			kernel_processPixelStage2_depth_channels.setArg(8, buf_x_table);
			kernel_processPixelStage2_depth_channels.setArg(9, buf_z_table);
			kernel_processPixelStage2_depth_channels.setArg(10, buf_depth);

			std::cout<<"processPixelStage2_depth_channels "<<err<<"\n";
			break;
		case 6:
			kernel_processPixelStage2_phase_channels3 = cl::Kernel(program, "processPixelStage2_phase_channels3", &err);
			kernel_processPixelStage2_phase_channels3.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_phase_channels3.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_phase_channels3.setArg(2, buf_phase_1);
			kernel_processPixelStage2_phase_channels3.setArg(3, buf_phase_2);
			kernel_processPixelStage2_phase_channels3.setArg(4, buf_phase_3);
			kernel_processPixelStage2_phase_channels3.setArg(5, buf_ir_sum);
			kernel_processPixelStage2_phase_channels3.setArg(6, buf_channels_1);
			kernel_processPixelStage2_phase_channels3.setArg(7, buf_channels_2);

			kernel_filter_channels = cl::Kernel(program, "filter_channels", &err);
			kernel_filter_channels.setArg(0, buf_channels_1);
			kernel_filter_channels.setArg(1, buf_channels_2);
			kernel_filter_channels.setArg(2, buf_channels_1_filtered);
			kernel_filter_channels.setArg(3, buf_channels_2_filtered);
			kernel_filter_channels.setArg(4, buf_gaussian_kernel);

			kernel_processPixelStage2_depth_channels3 = cl::Kernel(program, "processPixelStage2_depth_channels3", &err);
			kernel_processPixelStage2_depth_channels3.setArg(0, buf_phase_1);
			kernel_processPixelStage2_depth_channels3.setArg(1, buf_phase_2);
			kernel_processPixelStage2_depth_channels3.setArg(2, buf_phase_3);
			kernel_processPixelStage2_depth_channels3.setArg(3, buf_channels_1_filtered);
			kernel_processPixelStage2_depth_channels3.setArg(4, buf_channels_2_filtered);
			kernel_processPixelStage2_depth_channels3.setArg(5, buf_x_table);
			kernel_processPixelStage2_depth_channels3.setArg(6, buf_z_table);
			kernel_processPixelStage2_depth_channels3.setArg(7, buf_depth);
			break;
		case 7:
			kernel_processPixelStage2_phase_channels = cl::Kernel(program, "processPixelStage2_phase_channels", &err);
			kernel_processPixelStage2_phase_channels.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_phase_channels.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_phase_channels.setArg(2, buf_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(3, buf_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(4, buf_ir_sum);
			kernel_processPixelStage2_phase_channels.setArg(5, buf_channels_1);
			kernel_processPixelStage2_phase_channels.setArg(6, buf_channels_2);
			kernel_processPixelStage2_phase_channels.setArg(7, buf_channels_3);
			kernel_processPixelStage2_phase_channels.setArg(8, buf_channels_4);
			kernel_processPixelStage2_phase_channels.setArg(9, buf_channels_1_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(10, buf_channels_2_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(11, buf_channels_3_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(12, buf_channels_4_phase_1);
			kernel_processPixelStage2_phase_channels.setArg(13, buf_channels_1_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(14, buf_channels_2_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(15, buf_channels_3_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(16, buf_channels_4_phase_2);
			kernel_processPixelStage2_phase_channels.setArg(17, buf_ir_camera_intrinsics);

			kernel_filter_channels = cl::Kernel(program, "filter_channels", &err);
			kernel_filter_channels.setArg(0, buf_channels_1);
			kernel_filter_channels.setArg(1, buf_channels_2);
			kernel_filter_channels.setArg(2, buf_channels_3);
			kernel_filter_channels.setArg(3, buf_channels_4);
			kernel_filter_channels.setArg(4, buf_gaussian_kernel);
			kernel_filter_channels.setArg(5, buf_phase_1);
			kernel_filter_channels.setArg(6, buf_phase_2);
			kernel_filter_channels.setArg(7, buf_channels_1_phase_1);
			kernel_filter_channels.setArg(8, buf_channels_2_phase_1);
			kernel_filter_channels.setArg(9, buf_channels_3_phase_1);
			kernel_filter_channels.setArg(10, buf_channels_4_phase_1);
			kernel_filter_channels.setArg(11, buf_channels_1_phase_2);
			kernel_filter_channels.setArg(12, buf_channels_2_phase_2);
			kernel_filter_channels.setArg(13, buf_channels_3_phase_2);
			kernel_filter_channels.setArg(14, buf_channels_4_phase_2);
			kernel_filter_channels.setArg(15, buf_x_table);
			kernel_filter_channels.setArg(16, buf_z_table);
			kernel_filter_channels.setArg(17, buf_depth);
			break;
	}
	
	if(undistort_ == 1)
	{
		kernel_undistort = cl::Kernel(program, "undistort", &err);
		kernel_undistort.setArg(0, buf_depth);
		kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
		kernel_undistort.setArg(2, buf_depth_undistorted);
	}

	cl::Event event5;
	float* gauss_kernel;
	createGaussianKernel(&gauss_kernel, params.channel_filt_size);
	queue.enqueueWriteBuffer(buf_gaussian_kernel, CL_FALSE, 0, buf_gaussian_kernel_size, gauss_kernel, NULL, &event5);
	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4;
  queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
  queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
  queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
  queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
	queue.enqueueWriteBuffer(buf_ir_camera_intrinsics, CL_FALSE, 0, buf_ir_camera_intrinsics_size, ir_camera_intrinsics, NULL, &event4);

  event0.wait();
  event1.wait();
  event2.wait();
  event3.wait();
	event4.wait();
	event5.wait();

  programInitialized = true;
	std::cout<<"OpenCL program initialized\n";
  return true;
}

void logkinect::OpenCLChannelsBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
{
	
	static int cnt = 0;
	unsigned int offset = 0;

  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	if(bilateral_filter_ == 1)
  	queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	switch(pipeline_)
	{
		case 5 :
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_filter_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_depth_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			if(undistort_ == 1)
			{
				queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
				queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			}
			else
				queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			break;	
		case 7 :
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_filter_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			if(undistort_ == 1)
			{
				queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
				queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			}
			else
				queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			break;
		case 6 :
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase_channels3, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_filter_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_depth_channels3, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			if(undistort_ == 1)
			{
				queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
				queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			}
			else
				queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			break;		
	}	
	queue.finish();
}

bool logkinect::OpenCLChannelsBufferProcessorImpl::readProgram(std::string &source) const
{
	switch(pipeline_)
	{
		case 5:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_channel_depth_packet_processor.cl");
			std::cout<<"load source: /home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_channel_depth_packet_processor.cl \n";
			break;
		case 6:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_channel_depth_packet_processor.cl");
			break;
		case 7:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_channel_depth_packet_processor2.cl");
			break;
	}
}

bool logkinect::OpenCLShortChannelsBufferProcessorImpl::initProgram()
{
	initInitProgram();
	std::cout<<"initializing OpenCL buffers"<<std::endl;
	std::cout<<"bilateral filter = "<<bilateral_filter_<<std::endl;
	if(bilateral_filter_ != 1)
		std::cout<<"no bilateral filter is used\n";


	buf_depth_size = image_size * sizeof(cl_float)*4;
	buf_ir_sum_size = image_size * sizeof(cl_float)*3;

	kernel_processPixelStage2_phase_depth = cl::Kernel(program, "processPixelStage2_phase_depth", &err);
	kernel_processPixelStage2_phase_depth.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
	kernel_processPixelStage2_phase_depth.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
	kernel_processPixelStage2_phase_depth.setArg(2, buf_x_table);
	kernel_processPixelStage2_phase_depth.setArg(3, buf_z_table);
	kernel_processPixelStage2_phase_depth.setArg(4, buf_depth);
	kernel_processPixelStage2_phase_depth.setArg(5, buf_ir_sum);

	if(undistort_ == 1)
	{
		kernel_undistort = cl::Kernel(program, "undistort", &err);
		kernel_undistort.setArg(0, buf_depth);
		kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
		kernel_undistort.setArg(2, buf_depth_undistorted);
	}
	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4;
  queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
  queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
  queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
  queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
	queue.enqueueWriteBuffer(buf_ir_camera_intrinsics, CL_FALSE, 0, buf_ir_camera_intrinsics_size, ir_camera_intrinsics, NULL, &event4);

  event0.wait();
  event1.wait();
  event2.wait();
  event3.wait();
	event4.wait();

  programInitialized = true;
	std::cout<<"OpenCL program initialized\n";
  return true;
}

void logkinect::OpenCLHypDepthBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
{
	
	static int cnt = 0;
	unsigned int offset = 0;

  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	if(bilateral_filter_ == 1)
  	queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase_depth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
	if(dev_mode_>0)
	{
		queue.enqueueReadBuffer(buf_ir_sum, CL_FALSE, 0, buf_ir_sum_size, ir_packet_.buffer, NULL, NULL);
	}

	queue.finish();
}

bool logkinect::OpenCLHypDepthBufferProcessorImpl::readProgram(std::string &source) const
{
	source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl");
	std::cout<<"load source: /home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl \n";

}




