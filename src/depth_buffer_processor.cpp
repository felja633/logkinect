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

    oss << " -D MIN_DEPTH=" << 0.5 * 1000.0f << "f";
    oss << " -D MAX_DEPTH=" << 18.75 * 1000.0f << "f";
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
  program.build(options.c_str());

	std::vector<cl::Device> devices = mContext.getInfo<CL_CONTEXT_DEVICES>();
  queue = cl::CommandQueue(mContext, devices[0], 0, &err);
  std::string str;
  program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &str);
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
	buf_ir_size = image_size * sizeof(cl_float);
  buf_a_filtered_size = image_size * sizeof(cl_float3);
  buf_b_filtered_size = image_size * sizeof(cl_float3);
  buf_edge_test_size = image_size * sizeof(cl_uchar);
  buf_depth_size = image_size * sizeof(cl_float);
	buf_ir_sum_size = image_size * sizeof(cl_float);
	buf_filtered_size = image_size * sizeof(cl_float);
	//
	buf_count_size = 1*sizeof(cl_uint);

 	std::cout<<"initializing OpenCL buffers"<<std::endl;
  buf_a = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_a_size, NULL, &err);
  buf_b = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_b_size, NULL, &err);
  buf_n = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_n_size, NULL, &err);
	buf_ir = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_ir_size, NULL, &err);
  buf_a_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_a_filtered_size, NULL, &err);
  buf_b_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_b_filtered_size, NULL, &err);
  buf_edge_test = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_edge_test_size, NULL, &err);
  buf_depth = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_filtered = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_filtered_size, NULL, &err);
	//
	buf_phase_1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_phase_2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_phase_3 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	//buf_phase = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_w1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_w2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_w3 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_conf1 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_conf2 = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_count = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_count_size, NULL, &err);
	buf_phase_prop_vertical = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_phase_prop_horizontal = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_cost_prop_vertical =  cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_cost_prop_horizontal =  cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
	buf_ir_sum = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_ir_sum_size, NULL, &err);
	buf_ir_camera_intrinsics = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_ir_camera_intrinsics_size, NULL, &err);
#if UNDISTORT == 1
	buf_depth_undistorted = cl::Buffer(mContext, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
#endif

	std::cout<<"initializing OpenCL kernels"<<std::endl;
  kernel_processPixelStage1 = cl::Kernel(program, "processPixelStage1", &err);
  kernel_processPixelStage1.setArg(0, buf_lut11to16);
  kernel_processPixelStage1.setArg(1, buf_z_table);
  kernel_processPixelStage1.setArg(2, buf_p0_table);
  kernel_processPixelStage1.setArg(3, buf_buffer);
  kernel_processPixelStage1.setArg(4, buf_a);
  kernel_processPixelStage1.setArg(5, buf_b);
  kernel_processPixelStage1.setArg(6, buf_n);
	kernel_processPixelStage1.setArg(7, buf_ir);

  kernel_filterPixelStage1 = cl::Kernel(program, "filterPixelStage1", &err);
  kernel_filterPixelStage1.setArg(0, buf_a);
  kernel_filterPixelStage1.setArg(1, buf_b);
  kernel_filterPixelStage1.setArg(2, buf_n);
  kernel_filterPixelStage1.setArg(3, buf_a_filtered);
  kernel_filterPixelStage1.setArg(4, buf_b_filtered);
  kernel_filterPixelStage1.setArg(5, buf_edge_test);

	kernel_processPixelStage2_fullmask = cl::Kernel(program, "processPixelStage2_fullmask", &err);
	kernel_processPixelStage2_fullmask.setArg(0, buf_a_filtered);
	kernel_processPixelStage2_fullmask.setArg(1, buf_b_filtered);
	kernel_processPixelStage2_fullmask.setArg(2, buf_x_table);
	kernel_processPixelStage2_fullmask.setArg(3, buf_z_table);
	kernel_processPixelStage2_fullmask.setArg(4, buf_depth);
	kernel_processPixelStage2_fullmask.setArg(5, buf_ir_sum);

	kernel_processPixelStage2_nomask = cl::Kernel(program, "processPixelStage2_nomask", &err);
	kernel_processPixelStage2_nomask.setArg(0, buf_a_filtered);
	kernel_processPixelStage2_nomask.setArg(1, buf_b_filtered);
	kernel_processPixelStage2_nomask.setArg(2, buf_x_table);
	kernel_processPixelStage2_nomask.setArg(3, buf_z_table);
	kernel_processPixelStage2_nomask.setArg(4, buf_depth);

#if PIPELINE == 3
	kernel_processPixelStage2_phase = cl::Kernel(program, "processPixelStage2_phase3", &err);
	kernel_processPixelStage2_phase.setArg(0, buf_a_filtered);
	kernel_processPixelStage2_phase.setArg(1, buf_b_filtered);
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
#else
	kernel_processPixelStage2_phase = cl::Kernel(program, "processPixelStage2_phase", &err);
	kernel_processPixelStage2_phase.setArg(0, buf_a_filtered);
	kernel_processPixelStage2_phase.setArg(1, buf_b_filtered);
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
#endif

	kernel_processPixelStage2_depth = cl::Kernel(program, "processPixelStage2_depth", &err);
	//kernel_processPixelStage2_depth.setArg(0, buf_phase);
	kernel_processPixelStage2_depth.setArg(0, buf_phase_prop_vertical);
	kernel_processPixelStage2_depth.setArg(1, buf_phase_prop_horizontal);
	kernel_processPixelStage2_depth.setArg(2, buf_cost_prop_vertical);
	kernel_processPixelStage2_depth.setArg(3, buf_cost_prop_horizontal);
	kernel_processPixelStage2_depth.setArg(4, buf_x_table);
	kernel_processPixelStage2_depth.setArg(5, buf_z_table);
	kernel_processPixelStage2_depth.setArg(6, buf_depth);

  kernel_filterPixelStage2 = cl::Kernel(program, "filterPixelStage2", &err);
  kernel_filterPixelStage2.setArg(0, buf_depth);
  kernel_filterPixelStage2.setArg(1, buf_ir_sum);
  kernel_filterPixelStage2.setArg(2, buf_edge_test);
  kernel_filterPixelStage2.setArg(3, buf_filtered);

#if UNDISTORT == 1
#if PIPELINE == 0
	kernel_undistort = cl::Kernel(program, "undistort", &err);
  kernel_undistort.setArg(0, buf_filtered);
  kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
  kernel_undistort.setArg(2, buf_depth_undistorted);
#else
	kernel_undistort = cl::Kernel(program, "undistort", &err);
  kernel_undistort.setArg(0, buf_depth);
  kernel_undistort.setArg(1, buf_ir_camera_intrinsics);
  kernel_undistort.setArg(2, buf_depth_undistorted);
#endif
#endif

	std::cout<<"writing OpenCL kernel buffers"<<std::endl;
  cl::Event event0, event1, event2, event3, event4, event5, event6, event7;
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

void logkinect::OpenCLDepthBufferProcessorImpl::run(const unsigned char *buffer, int buffer_length)
{
	/*std::vector<cl::Event> eventWrite(1), eventPPS1(1), eventFPS1(1), eventPPS2(1), eventPPS2P(1), eventPPS2D(1), eventFPS2(1);
	std::vector< std::vector<cl::Event> > eventpropagatevertical, eventpropagatehorizontal, eventWritecountvertical, eventWritecounthorizontal;
  cl::Event event1;

  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, &eventWrite[0]);

  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventWrite, &eventPPS1[0]);
  queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventPPS1, &eventFPS1[0]);
	//new algorithm
	queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventFPS1, &eventPPS2P[0]);
	
	unsigned int offset = 0;
	for(unsigned int count = offset; count < 256; count++)
	{

		std::vector<cl::Event> tmp_event(1), tmp_write(1);
		eventpropagatevertical.push_back(tmp_event);
		eventWritecountvertical.push_back(tmp_write);
	}
	for(unsigned int count = offset; count < 212; count++)
	{
		std::vector<cl::Event> tmp_event(1), tmp_write(1);
		eventpropagatehorizontal.push_back(tmp_event);
		eventWritecounthorizontal.push_back(tmp_write);
	}
	for(unsigned int count = offset; count < 256; count++)
	{
		if(count == offset)
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, &eventPPS2P, &(eventWritecountvertical[count-offset][0]));
		}
		else
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, &(eventpropagatevertical[count-offset-1]), &(eventWritecountvertical[count-offset][0]));
		}
		queue.enqueueNDRangeKernel(kernel_propagate_vertical, cl::NullRange, 2*424, cl::NullRange, &(eventWritecountvertical[count-offset]), &(eventpropagatevertical[count-offset][0]));
	}

	for(unsigned int count = offset; count < 212; count++)
	{
		if(count == offset)
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, &(eventpropagatevertical.back()), &(eventWritecounthorizontal[count-offset][0]));
		}
		else
		{
			queue.enqueueWriteBuffer(buf_count, CL_FALSE, 0, buf_count_size, &count, &(eventpropagatehorizontal[count-offset-1]), &(eventWritecounthorizontal[count-offset][0]));
		}
		queue.enqueueNDRangeKernel(kernel_propagate_horizontal, cl::NullRange, 2*512, cl::NullRange, &(eventWritecounthorizontal[count-offset]), &(eventpropagatehorizontal[count-offset][0]));
	}		

	queue.enqueueNDRangeKernel(kernel_processPixelStage2_depth, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &(eventpropagatehorizontal.back()), &eventPPS2D[0]);

	//end new algorithm

  queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, &eventPPS2D, &event1);
  event1.wait();*/

	static int cnt = 0;
	//std::vector<cl::Event> eventASDF(1), read(1);

		
  queue.enqueueWriteBuffer(buf_buffer, CL_FALSE, 0, buf_buffer_size, buffer, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
  queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

#if (PIPELINE == 1) || (PIPELINE == 3)
	queue.enqueueNDRangeKernel(kernel_processPixelStage2_phase, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

	unsigned int offset = 0;
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
#if UNDISTORT == 1
	queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
#else
	queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
#endif //undistort

#else  //pipeline 1 or 3
#if PIPELINE == 0
	queue.enqueueNDRangeKernel(kernel_processPixelStage2_fullmask, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
#if UNDISTORT == 1
	queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
#else
	queue.enqueueReadBuffer(buf_filtered, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
#endif //undistort

#else //pipeline ==2
	queue.enqueueNDRangeKernel(kernel_processPixelStage2_nomask, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);

#if UNDISTORT == 1
	queue.enqueueNDRangeKernel(kernel_undistort,cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
	queue.enqueueReadBuffer(buf_depth_undistorted, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
#else
	queue.enqueueReadBuffer(buf_depth, CL_FALSE, 0, buf_depth_size, packet_.buffer, NULL, NULL);
#endif //undistort pipeline == 2

#endif //pipeline == 0

#endif //pipeline == 1 or 3

  //queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, NULL, NULL);
  queue.finish();

}

bool logkinect::OpenCLDepthBufferProcessorImpl::readProgram(std::string &source) const
{
  source = loadCLSource("/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/src/opencl_depth_packet_processor.cl");
  return !source.empty();
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
	packet_.buffer = new unsigned char[packet_.width*packet_.height*packet_.bytes_per_element];
}
logkinect::OpenCLDepthBufferProcessor::~OpenCLDepthBufferProcessor()
{
  delete impl_;
}

logkinect::OpenCLDepthBufferProcessor::OpenCLDepthBufferProcessor(unsigned char *p0_tables_buffer, size_t p0_tables_buffer_length, double* ir_intr, double* rgb_intr, double* rotation, double* translation)
{

  impl_ = new OpenCLDepthBufferProcessorImpl();
  impl_->programInitialized = false;

  loadXTableFromFile("kinect_parameters/xTable.bin");
  loadZTableFromFile("kinect_parameters/zTable.bin");
  load11To16LutFromFile("kinect_parameters/11to16.bin");

  loadP0TablesFromCommandResponse(p0_tables_buffer, p0_tables_buffer_length);
	setIntrinsics(ir_intr, rgb_intr, rotation, translation);

	if(!impl_->initProgram())
		std::cout<<"OpenCLDepthBufferProcessor ERROR: FAILED TO INIT CL PROGRAM \n";

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
		std::cout<<"impl_->ir_camera_intrinsics["<<i<<"] = "<<impl_->ir_camera_intrinsics[i]<<std::endl;
		impl_->rgb_camera_intrinsics[i] = static_cast<cl_float>(rgb_camera_intrinsics[i]);
	}

	for(unsigned int i = 0; i < 3; i++) {
		impl_->rel_trans[i] = static_cast<cl_float>(trans[i]);
	}

	float* rot = new float[3];
	rot[0] = static_cast<float>(rot_rod[0]);
	rot[1] = static_cast<float>(rot_rod[1]);
	rot[2] = static_cast<float>(rot_rod[2]);

	float theta = rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2];
	rot[0] = rot[0]/theta;
	rot[1] = rot[1]/theta;
	rot[2] = rot[2]/theta;

	impl_->rel_rot[0] = (cl_float)(1*cos(theta) + (1-cos(theta))*rot[0]*rot[0]+sin(theta)*0);
	impl_->rel_rot[1] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[0]*rot[1]+sin(theta)*(-rot[2]));
	impl_->rel_rot[2] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[0]*rot[2]+sin(theta)*rot[1]);

	impl_->rel_rot[3] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[1]*rot[0]+sin(theta)*rot[2]);
	impl_->rel_rot[4] = (cl_float)(1*cos(theta) + (1-cos(theta))*rot[1]*rot[1]+sin(theta)*0);
	impl_->rel_rot[5] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[1]*rot[2]+sin(theta)*(-rot[0]));

	impl_->rel_rot[6] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[2]*rot[0]+sin(theta)*(-rot[1]));
	impl_->rel_rot[7] = (cl_float)(0*cos(theta) + (1-cos(theta))*rot[2]*rot[1]+sin(theta)*rot[0]);
	impl_->rel_rot[8] = (cl_float)(1*cos(theta) + (1-cos(theta))*rot[2]*rot[2]+sin(theta)*0);
}

float* logkinect::OpenCLDepthBufferProcessor::getIrCameraParams()
{
	return impl_->ir_camera_intrinsics;
}
void logkinect::OpenCLDepthBufferProcessor::readResult(logkinect::Depth_Packet& out)
{
	out = impl_->packet_;
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

  min_depth = 0.0f;
  max_depth = 1500.0f;//1850000.0f;
}


