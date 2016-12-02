/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */
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

//#define M_PI_ 3.14159265359

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

static void saveShortArrayToFileBin(short* pts, int len, std::string name)
{
	std::ofstream fout(name.c_str(), std::ofstream::binary);

 	if(fout.is_open())
	{
		//for(int i = 0; i < len; i++)
		//	fout<<pts[i];
		fout.seekp(0,std::ofstream::end);
		fout.write((char*)pts,len*sizeof(short));
		fout.close();
	}
	else
	{
		std::cout<<"unable to open file\n";
	}
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

		oss << " -D KDE_SIGMA_SQR="<<(9.0f/(float)params.num_channels*1.1f)*(9.0f/(float)params.num_channels*1.1f);
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

bool logkinect::OpenCLDepthBufferProcessorImpl::initProgram()
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
	buf_rgb_camera_intrinsics_size = sizeof(cl_float)*7;
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
	buf_o_size = image_size * sizeof(cl_float3);
	buf_o_filt_size = image_size * sizeof(cl_float)*3;
if(pipeline_==4)
	buf_ir_size = image_size * sizeof(cl_float)*6;
else
	buf_ir_size = image_size * sizeof(cl_float);

  buf_a_filtered_size = image_size * sizeof(cl_float3);
  buf_b_filtered_size = image_size * sizeof(cl_float3);
  buf_edge_test_size = image_size * sizeof(cl_uchar);

if(pipeline_ == 4)
  buf_depth_size = image_size * sizeof(cl_float)*4;
else
	buf_depth_size = 2*image_size * sizeof(cl_float);
if(pipeline_==4)
	buf_ir_sum_size = image_size * sizeof(cl_float)*3;
else
	buf_ir_sum_size = image_size * sizeof(cl_float);

	buf_filtered_size = 2*image_size * sizeof(cl_float);
	//
	buf_count_size = 1*sizeof(cl_uint);

	buf_phase_size = image_size * sizeof(cl_float);
	
	buf_rgb_index_size = 2*image_size*sizeof(cl_int);

 	std::cout<<"initializing OpenCL buffers"<<std::endl;
	std::cout<<"bilateral filter = "<<bilateral_filter_<<std::endl;
	if(bilateral_filter_ != 1)
		std::cout<<"no bilateral filter is used\n";

  buf_a = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_a_size, NULL, &err);
  buf_b = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_b_size, NULL, &err);
  buf_n = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_n_size, NULL, &err);
	buf_o = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_o_size, NULL, &err);
	buf_o_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_o_filt_size, NULL, &err);
	buf_ir = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_ir_size, NULL, &err);
  buf_a_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_a_filtered_size, NULL, &err);
  buf_b_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_b_filtered_size, NULL, &err);
  buf_edge_test = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_edge_test_size, NULL, &err);
  buf_depth = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_filtered_size, NULL, &err);
	buf_undist_map = cl::Buffer(mContext, CL_READ_WRITE_CACHE, 2*buf_phase_size, NULL, &err);
	buf_rgb_index = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_rgb_index_size, NULL, &err);
	//
	buf_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	buf_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	buf_phase_3 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	//buf_phase = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);

	buf_gaussian_kernel_size = (2*params.channel_filt_size+1)*sizeof(cl_float);
	buf_gaussian_kernel = cl::Buffer(mContext, CL_READ_ONLY_CACHE, buf_gaussian_kernel_size, NULL, &err);
	buf_channels_size = image_size * sizeof(cl_float16);
	buf_halfchannels_size = image_size * sizeof(cl_float16)/2;
	buf_shortchannels_size = image_size * sizeof(cl_ushort16);

	
	buf_channels_1_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	buf_channels_2_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);

	if(pipeline_ == 8)
	{
		buf_channels_1_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_halfchannels_size, NULL, &err);
		buf_channels_2_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_halfchannels_size, NULL, &err);
		buf_channels_1_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_halfchannels_size, NULL, &err);
		buf_channels_2_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_halfchannels_size, NULL, &err);
		buf_channels_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_halfchannels_size, NULL, &err);
		buf_channels_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_halfchannels_size, NULL, &err);
	}
	else if(pipeline_ == 9)
	{
		buf_channels_1_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_shortchannels_size, NULL, &err);
		buf_channels_2_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_shortchannels_size, NULL, &err);
		buf_channels_1_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_shortchannels_size, NULL, &err);
		buf_channels_2_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_shortchannels_size, NULL, &err);
		buf_channels_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_shortchannels_size, NULL, &err);
		buf_channels_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_shortchannels_size, NULL, &err);
	}
	else
	{
		buf_channels_1_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_2_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_1_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_2_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
		buf_channels_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_channels_size, NULL, &err);
	}

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

	//if(pipeline_ == 10)
	//{
	//	buf_kde_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	//	buf_kde_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_phase_size, NULL, &err);
	//}

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

	buf_ir_sum = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_ir_sum_size, NULL, &err);
	buf_ir_camera_intrinsics = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_ir_camera_intrinsics_size, NULL, &err);
	buf_rgb_camera_intrinsics = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_rgb_camera_intrinsics_size, NULL, &err);
	buf_rel_trans = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_rel_trans_size, NULL, &err);
	buf_rel_rot = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_rel_rot_size, NULL, &err);
	buf_depth_undistorted = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);

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

	switch(pipeline_)
	{
		case 0:
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

			std::cout<<"filterPixelStage2 kernel "<<err<<"\n";

			kernel_mapColorToDepth = cl::Kernel(program, "mapColorToDepth",&err);
			kernel_mapColorToDepth.setArg(0, buf_filtered);
			kernel_mapColorToDepth.setArg(1, buf_undist_map);
			kernel_mapColorToDepth.setArg(2, buf_rgb_camera_intrinsics);
			kernel_mapColorToDepth.setArg(3, buf_rel_rot);
			kernel_mapColorToDepth.setArg(4, buf_rel_trans);
			kernel_mapColorToDepth.setArg(5, buf_rgb_index);
			break;
		case 1:
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
			break;

		case 2:
			kernel_processPixelStage2_nomask = cl::Kernel(program, "processPixelStage2_nomask", &err);
			kernel_processPixelStage2_nomask.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_nomask.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_nomask.setArg(2, buf_x_table);
			kernel_processPixelStage2_nomask.setArg(3, buf_z_table);
			kernel_processPixelStage2_nomask.setArg(4, buf_depth);
			kernel_processPixelStage2_nomask.setArg(5, buf_ir_sum);

			kernel_mapColorToDepth = cl::Kernel(program, "mapColorToDepth",&err);
			kernel_mapColorToDepth.setArg(0, buf_depth);
			kernel_mapColorToDepth.setArg(1, buf_undist_map);
			kernel_mapColorToDepth.setArg(2, buf_rgb_camera_intrinsics);
			kernel_mapColorToDepth.setArg(3, buf_rel_rot);
			kernel_mapColorToDepth.setArg(4, buf_rel_trans);
			kernel_mapColorToDepth.setArg(5, buf_rgb_index);
			break;
		case 3:
			kernel_processPixelStage2_phase = cl::Kernel(program, "processPixelStage2_phase3", &err);
			kernel_processPixelStage2_phase.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_phase.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_phase.setArg(2, buf_phase_1);
			kernel_processPixelStage2_phase.setArg(3, buf_phase_2);
			kernel_processPixelStage2_phase.setArg(4, buf_phase_3);
			kernel_processPixelStage2_phase.setArg(5, buf_phase_prop_vertical);
			kernel_processPixelStage2_phase.setArg(6, buf_phase_prop_horizontal);
			kernel_processPixelStage2_phase.setArg(7, buf_w1);
			kernel_processPixelStage2_phase.setArg(8, buf_w2);
			kernel_processPixelStage2_phase.setArg(9, buf_w3);
			kernel_processPixelStage2_phase.setArg(10, buf_cost_prop_vertical);
			kernel_processPixelStage2_phase.setArg(11, buf_cost_prop_horizontal);
			kernel_processPixelStage2_phase.setArg(12, buf_conf1);
			kernel_processPixelStage2_phase.setArg(13, buf_conf2);
			kernel_processPixelStage2_phase.setArg(14, buf_ir_sum);

			kernel_propagate_vertical = cl::Kernel(program, "propagateVertical3", &err);
			kernel_propagate_vertical.setArg(0, buf_phase_1);
			kernel_propagate_vertical.setArg(1, buf_phase_2);
			kernel_propagate_vertical.setArg(2, buf_phase_3);
			kernel_propagate_vertical.setArg(3, buf_w1);
			kernel_propagate_vertical.setArg(4, buf_w2);
			kernel_propagate_vertical.setArg(5, buf_w3);
			kernel_propagate_vertical.setArg(6, buf_conf1);
			kernel_propagate_vertical.setArg(7, buf_conf2);
			kernel_propagate_vertical.setArg(8, buf_count);
			kernel_propagate_vertical.setArg(9, buf_phase_prop_vertical);
			kernel_propagate_vertical.setArg(10, buf_cost_prop_vertical);

			kernel_propagate_horizontal = cl::Kernel(program, "propagateHorizontal3", &err);
			kernel_propagate_horizontal.setArg(0, buf_phase_1);
			kernel_propagate_horizontal.setArg(1, buf_phase_2);
			kernel_propagate_horizontal.setArg(2, buf_phase_3);
			kernel_propagate_horizontal.setArg(3, buf_w1);
			kernel_propagate_horizontal.setArg(4, buf_w2);
			kernel_propagate_horizontal.setArg(5, buf_w3);
			kernel_propagate_horizontal.setArg(6, buf_conf1);
			kernel_propagate_horizontal.setArg(7, buf_conf2);
			kernel_propagate_horizontal.setArg(8, buf_count);
			kernel_propagate_horizontal.setArg(9, buf_phase_prop_horizontal);
			kernel_propagate_horizontal.setArg(10, buf_cost_prop_horizontal);

			kernel_processPixelStage2_depth = cl::Kernel(program, "processPixelStage2_depth", &err);
			//kernel_processPixelStage2_depth.setArg(0, buf_phase);
			kernel_processPixelStage2_depth.setArg(0, buf_phase_prop_vertical);
			kernel_processPixelStage2_depth.setArg(1, buf_phase_prop_horizontal);
			kernel_processPixelStage2_depth.setArg(2, buf_cost_prop_vertical);
			kernel_processPixelStage2_depth.setArg(3, buf_cost_prop_horizontal);
			kernel_processPixelStage2_depth.setArg(4, buf_x_table);
			kernel_processPixelStage2_depth.setArg(5, buf_z_table);
			kernel_processPixelStage2_depth.setArg(6, buf_depth);

			std::cout<<"propagateHorizontal3 kernel \n";
			break;
		case 4:
	
			kernel_processPixelStage2_phase_depth = cl::Kernel(program, "processPixelStage2_phase_depth", &err);
			kernel_processPixelStage2_phase_depth.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_phase_depth.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_phase_depth.setArg(2, buf_x_table);
			kernel_processPixelStage2_phase_depth.setArg(3, buf_z_table);
			kernel_processPixelStage2_phase_depth.setArg(4, buf_depth);
			kernel_processPixelStage2_phase_depth.setArg(5, buf_ir_sum);

			break;

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
		case 8:
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
		case 9:
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
		case 10:

			kernel_processPixelStage2_phase = cl::Kernel(program, "processPixelStage2_phase", &err);
			kernel_processPixelStage2_phase.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_phase.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_phase.setArg(2, buf_phase_1);
			kernel_processPixelStage2_phase.setArg(3, buf_phase_2);
			kernel_processPixelStage2_phase.setArg(4, buf_w1);
			kernel_processPixelStage2_phase.setArg(5, buf_w2);
			kernel_processPixelStage2_phase.setArg(6, buf_ir_sum);

			kernel_filter_channels = cl::Kernel(program, "filter_kde", &err);
			kernel_filter_channels.setArg(0, buf_phase_1);
			kernel_filter_channels.setArg(1, buf_phase_2);
			kernel_filter_channels.setArg(2, buf_w1);
			kernel_filter_channels.setArg(3, buf_w2);
			//kernel_filter_channels.setArg(4, buf_kde_1);
			//kernel_filter_channels.setArg(5, buf_kde_2);
			kernel_filter_channels.setArg(4, buf_gaussian_kernel);
			kernel_filter_channels.setArg(5, buf_x_table);
			kernel_filter_channels.setArg(6, buf_z_table);
			kernel_filter_channels.setArg(7, buf_depth);
			kernel_filter_channels.setArg(8, buf_ir_sum);
			/*kernel_processPixelStage2_depth = cl::Kernel(program, "processPixelStage2_depth", &err);
			kernel_processPixelStage2_depth.setArg(0, buf_phase_1);
			kernel_processPixelStage2_depth.setArg(1, buf_phase_2);
			kernel_processPixelStage2_depth.setArg(2, buf_kde_1);
			kernel_processPixelStage2_depth.setArg(3, buf_kde_2);
			kernel_processPixelStage2_depth.setArg(4, buf_x_table);
			kernel_processPixelStage2_depth.setArg(5, buf_z_table);
			kernel_processPixelStage2_depth.setArg(6, buf_depth);*/

		  kernel_filterPixelStage2 = cl::Kernel(program, "filterPixelStage2", &err);
			kernel_filterPixelStage2.setArg(0, buf_depth);
			kernel_filterPixelStage2.setArg(1, buf_ir_sum);
			kernel_filterPixelStage2.setArg(2, buf_edge_test);
			kernel_filterPixelStage2.setArg(3, buf_filtered);

			kernel_mapColorToDepth = cl::Kernel(program, "mapColorToDepth",&err);
			kernel_mapColorToDepth.setArg(0, buf_depth);
			kernel_mapColorToDepth.setArg(1, buf_undist_map);
			kernel_mapColorToDepth.setArg(2, buf_rgb_camera_intrinsics);
			kernel_mapColorToDepth.setArg(3, buf_rel_rot);
			kernel_mapColorToDepth.setArg(4, buf_rel_trans);
			kernel_mapColorToDepth.setArg(5, buf_rgb_index);
			std::cout<<"processPixelStage2_kde "<<err<<"\n";
			break;
		case 11:
			kernel_processPixelStage2_phase = cl::Kernel(program, "processPixelStage2_phase3", &err);
			kernel_processPixelStage2_phase.setArg(0, bilateral_filter_ == 1? buf_a_filtered: buf_a);
			kernel_processPixelStage2_phase.setArg(1, bilateral_filter_ == 1? buf_b_filtered: buf_b);
			kernel_processPixelStage2_phase.setArg(2, buf_phase_1);
			kernel_processPixelStage2_phase.setArg(3, buf_phase_2);
			kernel_processPixelStage2_phase.setArg(4, buf_phase_3);
			kernel_processPixelStage2_phase.setArg(5, buf_w1);
			kernel_processPixelStage2_phase.setArg(6, buf_w2);
			kernel_processPixelStage2_phase.setArg(7, buf_w3);
			kernel_processPixelStage2_phase.setArg(8, buf_ir_sum);

			kernel_filter_channels = cl::Kernel(program, "filter_kde3", &err);
			kernel_filter_channels.setArg(0, buf_phase_1);
			kernel_filter_channels.setArg(1, buf_phase_2);
			kernel_filter_channels.setArg(2, buf_phase_3);
			kernel_filter_channels.setArg(3, buf_w1);
			kernel_filter_channels.setArg(4, buf_w2);
			kernel_filter_channels.setArg(5, buf_w3);
			//kernel_filter_channels.setArg(4, buf_kde_1);
			//kernel_filter_channels.setArg(5, buf_kde_2);
			kernel_filter_channels.setArg(6, buf_gaussian_kernel);
			kernel_filter_channels.setArg(7, buf_x_table);
			kernel_filter_channels.setArg(8, buf_z_table);
			kernel_filter_channels.setArg(9, buf_depth);

			kernel_mapColorToDepth = cl::Kernel(program, "mapColorToDepth",&err);
			kernel_mapColorToDepth.setArg(0, buf_depth);
			kernel_mapColorToDepth.setArg(1, buf_undist_map);
			kernel_mapColorToDepth.setArg(2, buf_rgb_camera_intrinsics);
			kernel_mapColorToDepth.setArg(3, buf_rel_rot);
			kernel_mapColorToDepth.setArg(4, buf_rel_trans);
			kernel_mapColorToDepth.setArg(5, buf_rgb_index);
		default:
			std::cout<<"pipeline finns inte \n";
	}

	cl::Event event5;
	float* gauss_kernel;
	createGaussianKernel(&gauss_kernel, params.channel_filt_size);
	queue.enqueueWriteBuffer(buf_gaussian_kernel, CL_FALSE, 0, buf_gaussian_kernel_size, gauss_kernel, NULL, &event5);
	/*kernel_filter_channels2 = cl::Kernel(program, "filter_channels_fast", &err);
	kernel_filter_channels2.setArg(0, buf_channels_1_filtered);
	kernel_filter_channels2.setArg(1, buf_channels_2_filtered);
	kernel_filter_channels2.setArg(2, buf_channels_1);
	kernel_filter_channels2.setArg(3, buf_channels_2);
	kernel_filter_channels2.setArg(4, buf_gaussian_kernel);*/

	std::cout<<"filterPixelStage2 kernel \n";
	if(undistort_ == 1)
	{
		if(pipeline_ == 0)
		{
			kernel_undistort = cl::Kernel(program, "undistort", &err);
			kernel_undistort.setArg(0, buf_filtered);
			kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
			kernel_undistort.setArg(2, buf_depth_undistorted);
		}
		else
		{
			kernel_undistort = cl::Kernel(program, "undistort", &err);
			kernel_undistort.setArg(0, buf_depth);
			kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
			kernel_undistort.setArg(2, buf_depth_undistorted);
		}
	}

	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4, event6, event7, event8, event9;
  queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
  queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
  queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
  queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
	queue.enqueueWriteBuffer(buf_ir_camera_intrinsics, CL_FALSE, 0, buf_ir_camera_intrinsics_size, ir_camera_intrinsics, NULL, &event4);
	queue.enqueueWriteBuffer(buf_rgb_camera_intrinsics, CL_FALSE, 0, buf_rgb_camera_intrinsics_size, rgb_camera_intrinsics, NULL, &event6);
	queue.enqueueWriteBuffer(buf_rel_rot, CL_FALSE, 0, buf_rel_rot_size, rel_rot, NULL, &event7);
	queue.enqueueWriteBuffer(buf_rel_trans, CL_FALSE, 0, buf_rel_trans_size, rel_trans, NULL, &event8);
	queue.enqueueWriteBuffer(buf_undist_map, CL_FALSE, 0, 2*buf_phase_size, undist_im, NULL, &event9);

  event0.wait();
  event1.wait();
  event2.wait();
  event3.wait();
	event4.wait();
	event5.wait();
	event6.wait();
	event7.wait();
	event8.wait();
	event9.wait();

	delete[] gauss_kernel;
  programInitialized = true;
	std::cout<<"OpenCL program initialized\n";
  return true;
}

void logkinect::OpenCLDepthBufferProcessorImpl::createGaussianKernel(float** kernel, int size)
{
	*kernel = new float[2*size+1];
	float sigma = 0.5f*(float)size;
	std::cout<<"gaussian kernel filt size = "<<size<<std::endl;
	std::cout<<"kernel = [";
	for(int i = -size; i <= size; i++)	
	{
		(*kernel)[i+size] = exp(-0.5f*i*i/(sigma*sigma)); 	
		std::cout<<(*kernel)[i+size]<<", "; 
	}
	std::cout<<"\n";
}

void logkinect::OpenCLDepthBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
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
		case 8:
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
		case 9:
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
		case 10:
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_filter_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			//queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			//queue.enqueueNDRangeKernel(kernel_processPixelStage2_depth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_mapColorToDepth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			if(undistort_ == 1)
			{
				queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
				queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			}
			else
				queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);

			queue.enqueueReadBuffer(buf_rgb_index, CL_FALSE, 0, buf_rgb_index_size, rgb_index_packet_.buffer, NULL, NULL);
			break;
		case 11:
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_filter_channels, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			//queue.enqueueNDRangeKernel(kernel_processPixelStage2_depth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueReadBuffer(buf_rgb_index, CL_FALSE, 0, buf_rgb_index_size, rgb_index_packet_.buffer, NULL, NULL);
			if(undistort_ == 1)
			{
				queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
				queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			}
			else
				queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);

			queue.enqueueNDRangeKernel(kernel_mapColorToDepth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
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
		case 1 :
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
			break;
		case 3:
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
			break;
		case 0:
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_fullmask, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_mapColorToDepth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			if(undistort_ == 1)
			{
				queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
				queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			}
			else
				queue.enqueueReadBuffer(buf_filtered, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);

			queue.enqueueReadBuffer(buf_rgb_index, CL_FALSE, 0, buf_rgb_index_size, rgb_index_packet_.buffer, NULL, NULL);
			break;
		case 2:
			std::cout<<"pipeline 2 processing \n";
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_nomask, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueNDRangeKernel(kernel_mapColorToDepth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			if(undistort_ == 1)
			{
				queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
				queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			}
			else
				queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);

			
			queue.enqueueReadBuffer(buf_rgb_index, CL_FALSE, 0, buf_rgb_index_size, rgb_index_packet_.buffer, NULL, NULL);
			break;
		case 4:
			queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase_depth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
			queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
			if(dev_mode_>0)
			{
				queue.enqueueReadBuffer(buf_ir_sum, CL_FALSE, 0, buf_ir_sum_size, ir_packet_.buffer, NULL, NULL);
				//queue.enqueueReadBuffer(buf_o_filtered, CL_FALSE, 0, buf_o_filt_size, ir_packet_.buffer+512*424*4*3, NULL, NULL);
			}
			break;
		default:
			std::cout<<"pipeline finns inte \n";
	}

  queue.finish();

}

bool logkinect::OpenCLDepthBufferProcessorImpl::readProgram(std::string &source) const
{
	std::cout<<"pipeline_ = "<<pipeline_<<std::endl;
	switch(pipeline_)
	{
		case 0:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_depth_packet_processor.cl");
			break;
		case 1:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl");
			break;
		case 2:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_depth_packet_processor.cl");
			break;
		case 3:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl");
			break;
		case 4:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_spatial_propagation_depth_packet_processor.cl");
			break;
		case 5:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_channel_depth_packet_processor.cl");
			break;
		case 6:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_channel_depth_packet_processor.cl");
			break;
		case 7:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_channel_depth_packet_processor2.cl");
			break;
		case 8:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_halfchannel_depth_packet_processor.cl");
			break;
		case 9:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_shortchannel_depth_packet_processor.cl");
			break;
		case 10:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_kde_depth_packet_processor.cl");
			break;
		case 11:
			source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_kde_depth_packet_processor.cl");
			break;
		default:
			std::cout<<"must set pipeline_\n";
	}
  
  return !source.empty();
}

void logkinect::OpenCLDepthBufferProcessorImpl::fill_trig_table(const P0TablesResponse *p0table)
{

	for(int r = 0; r < 424; ++r)
	{
		cl_float3 *it = &p0_table[r * 512];
		const uint16_t *it0 = &p0table->p0table0[r * 512];
		const uint16_t *it1 = &p0table->p0table1[r * 512];
		const uint16_t *it2 = &p0table->p0table2[r * 512];
		//std::cout<<r<<std::endl;
		for(int c = 0; c < 512; ++c, ++it, ++it0, ++it1, ++it2)
		{
			it->s[0] = -((float) * it0) * 0.000031 * M_PI;
			it->s[1] = -((float) * it1) * 0.000031 * M_PI;
			it->s[2] = -((float) * it2) * 0.000031 * M_PI;
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
	{
		packet_.buffer = new unsigned char[packet_.width*packet_.height*packet_.bytes_per_element*2];		
	}

}
logkinect::OpenCLDepthBufferProcessor::~OpenCLDepthBufferProcessor()
{
  delete impl_;
}

logkinect::OpenCLDepthBufferProcessor::OpenCLDepthBufferProcessor(unsigned char *p0_tables_buffer, size_t p0_tables_buffer_length, double* ir_intr, double* rgb_intr, double* rotation, double* translation, int pipeline, int undistort, int bilateral_filter, const int dev_mode)
{

	std::cout<<"logkinect::OpenCLDepthBufferProcessor::OpenCLDepthBufferProcessor pipeline = "<<pipeline<<std::endl;
	std::cout<<"logkinect::OpenCLDepthBufferProcessor::OpenCLDepthBufferProcessor bilateral_filter = "<<bilateral_filter<<std::endl;
	
  impl_ = new OpenCLDepthBufferProcessorImpl(pipeline, undistort, bilateral_filter,dev_mode);
	
	
  impl_->programInitialized = false;

  const double scaling_factor = 8192;
  const double unambigious_dist = 6250.0/3;
  size_t divergence = 0;

	double fx = ir_intr[0];
	double fy = ir_intr[1];
	double cx = ir_intr[2];
	double cy = ir_intr[3];

  for (size_t i = 0; i < 512*424; i++)
  {
    size_t xi = i % 512;
    size_t yi = i / 512;
    double xd = (xi + 0.5 - cx)/fx;
    double yd = (yi + 0.5 - cy)/fy;
    double xu, yu;
    divergence += !undistort_image(xd, yd, xu, yu,ir_intr);
    impl_->x_table[i] = (float)scaling_factor*xu;
    impl_->z_table[i] = (float)unambigious_dist/sqrt(xu*xu + yu*yu + 1);
		impl_->undist_im[2*i] = xu;
		impl_->undist_im[2*i+1] = yu;
  }

  if (divergence > 0)
    std::cout << divergence << " pixels in x/ztable have incorrect undistortion.";

  short y = 0;
  for (int x = 0; x < 1024; x++)
  {
    unsigned inc = 1 << (x/128 - (x>=128));
    impl_->lut11to16[x] = (float)y;
    impl_->lut11to16[1024 + x] = (float)-y;
    y += inc;
  }
  impl_->lut11to16[1024] = 32767;

/*
  loadXTableFromFile("kinect_parameters/xTable.bin");
  loadZTableFromFile("kinect_parameters/zTable.bin");
  load11To16LutFromFile("kinect_parameters/11to16.bin");*/

  loadP0TablesFromCommandResponse(p0_tables_buffer, p0_tables_buffer_length);
	setIntrinsics(ir_intr, rgb_intr, rotation, translation);
	std::cout<<"pipeline = "<<impl_->pipeline_<<std::endl;
	if(!impl_->initProgram())
		std::cout<<"OpenCLDepthBufferProcessor ERROR: FAILED TO INIT CL PROGRAM \n";

	saveFloatArrayToFileBin(impl_->z_table, 512*424, "z_table.bin");
	saveFloatArrayToFileBin(impl_->x_table, 512*424, "x_table.bin");
	saveFloatArrayToFileBin(impl_->undist_im, 512*424*2, "undist_im.bin");
	saveShortArrayToFileBin(impl_->lut11to16, 2048, "lut11to16.bin");
	std::cout<<"init done\n";
}

void logkinect::OpenCLDepthBufferProcessor::loadP0TablesFromCommandResponse(unsigned char *buffer, size_t buffer_length)
{
  P0TablesResponse *p0table = (P0TablesResponse *)buffer;

  if(buffer_length < sizeof(P0TablesResponse))
  {
    std::cerr << "P0Table response too short!" << std::endl;
    return;
  }

  impl_->fill_trig_table(p0table);
	float* tabl = new float[512*424*3];
	
	for(unsigned int i = 0; i < 512*424; i++)
	{
		tabl[i] = impl_->p0_table[i].s[0];
		tabl[i+512*424] = impl_->p0_table[i].s[1];
		tabl[i+2*512*424] = impl_->p0_table[i].s[2];
	}
	saveFloatArrayToFileBin(tabl, 512*424*3, "p0_table.bin");
	delete[] tabl;
 // std::cout << "trig tables filled" << std::endl;
}


void logkinect::OpenCLDepthBufferProcessor::loadXTableFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->x_table, impl_->image_size * sizeof(float)))
  {
    std::cerr << "could not load x table from: " << filename << std::endl;
  }
}

void logkinect::OpenCLDepthBufferProcessor::loadZTableFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->z_table, impl_->image_size * sizeof(float)))
  {
    std::cerr <<"could not load z table from: " << filename << std::endl;
  }
}

void logkinect::OpenCLDepthBufferProcessor::load11To16LutFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->lut11to16, 2048 * sizeof(cl_ushort)))
  {
    std::cerr << "could not load lut table from: " << filename << std::endl;
  }

}

void logkinect::OpenCLDepthBufferProcessor::process(const unsigned char *buffer, int buffer_length)
{
	//if(!impl_->programInitialized)
  if(!impl_->programInitialized)
  {
    std::cerr << "could not initialize OpenCLDepthBufferProcessor" << std::endl;
    return;
  }
	impl_->initNewPacket();
  impl_->run(buffer, buffer_length);

}

void logkinect::OpenCLDepthBufferProcessor::setIntrinsics(double* ir_camera_intrinsics, double* rgb_camera_intrinsics, double* rot_rod, double* trans)
{
	std::cout<<"OpenCLDepthBufferProcessor::setIntrinsics \n";
	for(unsigned int i = 0; i < 7; i++) {
		impl_->ir_camera_intrinsics[i] = static_cast<cl_float>(ir_camera_intrinsics[i]);
		
		impl_->rgb_camera_intrinsics[i] = static_cast<cl_float>(rgb_camera_intrinsics[i]);
		std::cout<<"impl_->rgb_camera_intrinsics["<<i<<"] = "<<impl_->rgb_camera_intrinsics[i]<<std::endl;
	}

	std::cout<<"rel trans: \n";
	for(unsigned int i = 0; i < 3; i++) {
		impl_->rel_trans[i] = static_cast<cl_float>(trans[i]);
		std::cout<<impl_->rel_trans[i]<<std::endl;
	}

	float* rot = new float[3];
	rot[0] = static_cast<float>(rot_rod[0]);
	rot[1] = static_cast<float>(rot_rod[1]);
	rot[2] = static_cast<float>(rot_rod[2]);

	float theta = rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2];
	rot[0] = rot[0]/theta;
	rot[1] = rot[1]/theta;
	rot[2] = rot[2]/theta;

	std::cout<<"rel rot: \n";
	impl_->rel_rot[0] = (cl_float)(1*cos(theta) + (1-cos(theta))*rot[0]*rot[0]+sin(theta)*0);
	impl_->rel_rot[1] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[0]*rot[1]+sin(theta)*(-rot[2]));
	impl_->rel_rot[2] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[0]*rot[2]+sin(theta)*rot[1]);

	impl_->rel_rot[3] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[1]*rot[0]+sin(theta)*rot[2]);
	impl_->rel_rot[4] = (cl_float)(1*cos(theta) + (1-cos(theta))*rot[1]*rot[1]+sin(theta)*0);
	impl_->rel_rot[5] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[1]*rot[2]+sin(theta)*(-rot[0]));

	impl_->rel_rot[6] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[2]*rot[0]+sin(theta)*(-rot[1]));
	impl_->rel_rot[7] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[2]*rot[1]+sin(theta)*rot[0]);
	impl_->rel_rot[8] = (cl_float)(1*cos(theta) + (1-cos(theta))*rot[2]*rot[2]+sin(theta)*0);

	for(unsigned int i = 0; i < 9; i++) {
		impl_->rel_trans[i] = static_cast<cl_float>(trans[i]);
		std::cout<<impl_->rel_rot[i]<<std::endl;
	}
	delete[] rot;
}

float* logkinect::OpenCLDepthBufferProcessor::getIrCameraParams()
{
	return impl_->ir_camera_intrinsics;
}
void logkinect::OpenCLDepthBufferProcessor::readResult(logkinect::Depth_Packet& out)
{
	out = impl_->packet_;
}

void logkinect::OpenCLDepthBufferProcessor::readPointCloud(float* point_cloud, int* rgb_indices)
{
	float* depth = (float*)impl_->packet_.buffer;
	int* inds = reinterpret_cast<int*>(impl_->rgb_index_packet_.buffer);
	unsigned int x, y, mirror_index;
	for(unsigned int i = 0; i < impl_->image_size; i++)
	{
		x =511 - (i % 512);
		y = i / 512;
		mirror_index = (512*y)+x;
		point_cloud[3*mirror_index] = depth[mirror_index]*(-impl_->undist_im[2*mirror_index]);
		point_cloud[3*mirror_index+1] = depth[mirror_index]*impl_->undist_im[2*mirror_index+1];
		point_cloud[3*mirror_index+2] = depth[mirror_index];
		//std::cout<<"x = "<<inds[2*i]<<" y = "<<inds[2*i+1]<<std::endl;
		rgb_indices[2*mirror_index] = inds[2*mirror_index]; 
		rgb_indices[2*mirror_index+1] = inds[2*mirror_index+1];
	}
}

void logkinect::OpenCLDepthBufferProcessor::readIrResult(logkinect::Depth_Packet& out)
{
	if(impl_->dev_mode_>0)
		out = impl_->ir_packet_;
}

void logkinect::OpenCLDepthBufferProcessor::distort(double x, double y, double &xd, double &yd, double* ir_intr) const
{
	double k1 = ir_intr[4];
	double k2 = ir_intr[5];
	double k3 = ir_intr[6];
	double p1 = 0.0;
	double p2 = 0.0;

  double x2 = x * x;
  double y2 = y * y;
  double r2 = x2 + y2;
  double xy = x * y;
  double kr = ((k3 * r2 + k2) * r2 + k1) * r2 + 1.0;
  xd = x*kr + p2*(r2 + 2*x2) + 2*p1*xy;
  yd = y*kr + p1*(r2 + 2*y2) + 2*p2*xy;
}

//The inverse of distort() using Newton's method
//Return true if converged correctly
//This function considers tangential distortion with double precision.
bool logkinect::OpenCLDepthBufferProcessor::undistort_image(double x, double y, double &xu, double &yu, double* ir_intr) const
{

	double k1 = ir_intr[4];
	double k2 = ir_intr[5];
	double k3 = ir_intr[6];
	double p1 = 0.0;
	double p2 = 0.0;

  double x0 = x;
  double y0 = y;

  double last_x = x;
  double last_y = y;
  const int max_iterations = 100;
  int iter;
  for (iter = 0; iter < max_iterations; iter++) {
    double x2 = x*x;
    double y2 = y*y;
    double x2y2 = x2 + y2;
    double x2y22 = x2y2*x2y2;
    double x2y23 = x2y2*x2y22;

    //Jacobian matrix
    double Ja = k3*x2y23 + (k2+6*k3*x2)*x2y22 + (k1+4*k2*x2)*x2y2 + 2*k1*x2 + 6*p2*x + 2*p1*y + 1;
    double Jb = 6*k3*x*y*x2y22 + 4*k2*x*y*x2y2 + 2*k1*x*y + 2*p1*x + 2*p2*y;
    double Jc = Jb;
    double Jd = k3*x2y23 + (k2+6*k3*y2)*x2y22 + (k1+4*k2*y2)*x2y2 + 2*k1*y2 + 2*p2*x + 6*p1*y + 1;

    //Inverse Jacobian
    double Jdet = 1/(Ja*Jd - Jb*Jc);
    double a = Jd*Jdet;
    double b = -Jb*Jdet;
    double c = -Jc*Jdet;
    double d = Ja*Jdet;

    double f, g;
    distort(x, y, f, g, ir_intr);
    f -= x0;
    g -= y0;

    x -= a*f + b*g;
    y -= c*f + d*g;
    const double eps = std::numeric_limits<double>::epsilon()*16;
    if (fabs(x - last_x) <= eps && fabs(y - last_y) <= eps)
      break;
    last_x = x;
    last_y = y;
  }
  xu = x;
  yu = y;
  return iter < max_iterations;
}


logkinect::Parameters::Parameters()
{
  ab_multiplier = 0.6666667f;
  ab_multiplier_per_frq[0] = 1.322581f;
  ab_multiplier_per_frq[1] = 1.0f;
  ab_multiplier_per_frq[2] = 1.612903f;
  ab_output_multiplier = 16.0f;

  phase_in_rad[0] = 0.0f;
  phase_in_rad[1] = 2.094395f;
  phase_in_rad[2] = 4.18879f;

  joint_bilateral_ab_threshold = 3.0f;
  joint_bilateral_max_edge = 2.5f;
  joint_bilateral_exp = 5.0f;

  gaussian_kernel[0] = 0.1069973f;
  gaussian_kernel[1] = 0.1131098f;
  gaussian_kernel[2] = 0.1069973f;
  gaussian_kernel[3] = 0.1131098f;
  gaussian_kernel[4] = 0.1195715f;
  gaussian_kernel[5] = 0.1131098f;
  gaussian_kernel[6] = 0.1069973f;
  gaussian_kernel[7] = 0.1131098f;
  gaussian_kernel[8] = 0.1069973f;

  phase_offset = 0.0f;
  unambigious_dist = 2083.333f;
  individual_ab_threshold  = 3.0f;
  ab_threshold = 10.0f;
  ab_confidence_slope = -0.5330578f;
  ab_confidence_offset = 0.7694894f;
  min_dealias_confidence = 0.3490659f;
  max_dealias_confidence = 0.6108653f;

  edge_ab_avg_min_value = 50.0f;
  edge_ab_std_dev_threshold = 0.05f;
  edge_close_delta_threshold = 50.0f;
  edge_far_delta_threshold = 30.0f;
  edge_max_delta_threshold = 100.0f;
  edge_avg_delta_threshold = 0.0f;
  max_edge_count  = 5.0f;

  min_depth = 0.5f;
  max_depth = 18.75f;//18.75f;

	num_channels = 64;
	channel_filt_size = 5;
	channel_confidence_scale = 2.0f;
	block_size_col = 8;
	block_size_row = 8;
}


