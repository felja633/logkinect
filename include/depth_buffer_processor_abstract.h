#include <CL/cl.hpp>

#ifndef DEPTH_BUFFER_PROCESSOR_H
#define DEPTH_BUFFER_PROCESSOR_H
namespace logkinect
{
struct Depth_Packet
{
	unsigned char* buffer;
	size_t lenght, bytes_per_element, height, width;
};

struct Parameters
{
    float ab_multiplier;
    float ab_multiplier_per_frq[3];
    float ab_output_multiplier;

    float phase_in_rad[3];

    float joint_bilateral_ab_threshold;
    float joint_bilateral_max_edge;
    float joint_bilateral_exp;
    float gaussian_kernel[9];

    float phase_offset;
    float unambigious_dist;
    float individual_ab_threshold;
    float ab_threshold;
    float ab_confidence_slope;
    float ab_confidence_offset;
    float min_dealias_confidence;
    float max_dealias_confidence;

    float edge_ab_avg_min_value;
    float edge_ab_std_dev_threshold;
    float edge_close_delta_threshold;
    float edge_far_delta_threshold;
    float edge_max_delta_threshold;
    float edge_avg_delta_threshold;
    float max_edge_count;

    float min_depth;
    float max_depth;

		unsigned int num_channels;
		unsigned int channel_filt_size;
		float channel_confidence_scale;
		unsigned int block_size_col;
		unsigned int block_size_row;

    Parameters();
};

struct P0TablesResponse
{
  uint32_t headersize;
  uint32_t unknown1;
  uint32_t unknown2;
  uint32_t tablesize;
  uint32_t unknown3;
  uint32_t unknown4;
  uint32_t unknown5;
  uint32_t unknown6;

  uint16_t unknown7;
  uint16_t p0table0[512*424]; // row[0] == row[511] == 0x2c9a
  uint16_t unknown8;

  uint16_t unknown9;
  uint16_t p0table1[512*424]; // row[0] == row[511] == 0x08ec
  uint16_t unknownA;

  uint16_t unknownB;
  uint16_t p0table2[512*424]; // row[0] == row[511] == 0x42e8
  uint16_t unknownC;

  uint8_t  unknownD[];
};

class OpenCLDepthBufferProcessorImpl
{

public:
	OpenCLDepthBufferProcessorImpl(int pipeline, int undistort, int bilateral_filter, const int dev_mode = -1, const int deviceId = -1) : deviceInitialized(false), programInitialized(false)
	{
		pipeline_ = pipeline;
		undistort_ = undistort;
		
		bilateral_filter_ = bilateral_filter;
		image_size = 512 * 424;

		lut11to16 = new cl_short[2048];
		x_table = new cl_float[image_size];
		z_table = new cl_float[image_size];
		undist_im = new cl_float[2*image_size];
		p0_table = new cl_float3[image_size];
		packet_.height = 424;
		packet_.width = 512;
		packet_.bytes_per_element = 4;
		dev_mode_ = dev_mode; 
		if(dev_mode>=0)
		{
			ir_packet_.height = 424;
			ir_packet_.width = 512;
			ir_packet_.bytes_per_element = 4;
		}
		//mContext = context;
		deviceInitialized = initDevice(deviceId);
		
		
	}

  ~OpenCLDepthBufferProcessorImpl();
	
	logkinect::Depth_Packet packet_;
	logkinect::Depth_Packet ir_packet_;

	int pipeline_, undistort_, dev_mode_, bilateral_filter_;	

  cl_short* lut11to16;
  cl_float* x_table;
  cl_float* z_table;
	cl_float* undist_im;
  cl_float3* p0_table;
  //libfreenect2::DepthBufferProcessor::Config config;
  logkinect::Parameters params;

	cl_float ir_camera_intrinsics[7];
	cl_float rgb_camera_intrinsics[7];
	cl_float rel_rot[9];
	cl_float rel_trans[3];

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  cl::Context mContext;
	cl::Device device;

  cl::Program program;
  cl::CommandQueue queue;



  cl::Kernel kernel_processPixelStage1;
  cl::Kernel kernel_filterPixelStage1;

	cl::Kernel kernel_undistort;

  size_t image_size;

  // Read only buffers
  size_t buf_lut11to16_size;
  size_t buf_p0_table_size;
  size_t buf_x_table_size;
  size_t buf_z_table_size;
  size_t buf_buffer_size;
	size_t buf_rel_rot_size;
	size_t buf_rel_trans_size;
	size_t buf_ir_camera_intrinsics_size;

  cl::Buffer buf_lut11to16;
  cl::Buffer buf_p0_table;
  cl::Buffer buf_x_table;
  cl::Buffer buf_z_table;
  cl::Buffer buf_buffer;

  // Read-Write buffers
  size_t buf_a_size;
  size_t buf_b_size;
  size_t buf_n_size;
	size_t buf_ir_size;
  size_t buf_a_filtered_size;
  size_t buf_b_filtered_size;
  size_t buf_edge_test_size;
  size_t buf_depth_size;
	size_t buf_phase_size;
  size_t buf_filtered_size;
	size_t buf_ir_sum_size;
	//
	size_t buf_count_size;
	size_t buf_rgb_index_size;

	size_t buf_o_size;
	size_t buf_o_filt_size;

  cl::Buffer buf_a;
  cl::Buffer buf_b;
  cl::Buffer buf_n;
  cl::Buffer buf_a_filtered;
  cl::Buffer buf_b_filtered;
  cl::Buffer buf_edge_test;
  cl::Buffer buf_depth;
	cl::Buffer buf_ir;
	cl::Buffer buf_ir_sum;
	cl::Buffer buf_o;
	cl::Buffer buf_o_filtered;

	cl::Buffer buf_phase_1;
	cl::Buffer buf_phase_2;

	cl::Buffer buf_gaussian_kernel;

	cl::Buffer buf_filtered;

  cl::Buffer buf_depth_undistorted;

	cl::Buffer buf_ir_camera_intrinsics;
	bool deviceInitialized;
  bool programInitialized;
  std::string sourceCode;

 	void generateOptions(std::string &options);
  bool initDevice(const int deviceId);
	void getDevices(const std::vector<cl::Platform> &platforms, std::vector<cl::Device> &devices);
	void listDevice(std::vector<cl::Device> &devices);
  bool selectDevice(std::vector<cl::Device> &devices, const int deviceId);
	bool initInitProgram();
  virtual bool initProgram() = 0;
  virtual void run(const unsigned char* buffer, int length) = 0;
  void fill_trig_table(const P0TablesResponse *p0table);
	void initNewPacket();

	virtual bool readProgram(std::string &source) const = 0;

};

class OpenCLLibfreenectFullmaskDepthBufferProcessorImpl : OpenCLDepthBufferProcessorImpl
{
	virtual bool initProgram();
	virtual void run(const unsigned char* buffer, int length);
	virtual bool readProgram(std::string &source) const;
	cl::Kernel kernel_processPixelStage2_fullmask;
  cl::Kernel kernel_filterPixelStage2;
};

class OpenCLLibfreenectNomaskDepthBufferProcessorImpl : OpenCLDepthBufferProcessorImpl
{
	virtual bool initProgram();
	virtual void run(const unsigned char* buffer, int length);
	virtual bool readProgram(std::string &source) const;
	cl::Kernel kernel_processPixelStage2_nomask;
};

class OpenCLHypDepthBufferProcessorImpl : OpenCLDepthBufferProcessorImpl
{
	virtual bool initProgram();
	virtual void run(const unsigned char* buffer, int length);
	virtual bool readProgram(std::string &source) const;
	cl::Kernel kernel_processPixelStage2_phase_depth;
};

class OpenCLSpatialPropagationDepthBufferProcessorImpl : OpenCLDepthBufferProcessorImpl
{
	virtual bool initProgram();
	virtual void run(const unsigned char* buffer, int length);
	virtual bool readProgram(std::string &source) const;

  cl::Kernel kernel_processPixelStage2_depth;
	cl::Kernel kernel_propagate_vertical;
  cl::Kernel kernel_propagate_horizontal;

	cl::Buffer buf_phase_1;
	cl::Buffer buf_phase_2;
	cl::Buffer buf_phase_3;
	cl::Buffer buf_w1;
	cl::Buffer buf_w2;
	cl::Buffer buf_w3;
	cl::Buffer buf_count;
	cl::Buffer buf_phase_prop_vertical;
	cl::Buffer buf_phase_prop_horizontal;
	cl::Buffer buf_cost_prop_vertical;
	cl::Buffer buf_cost_prop_horizontal;
};

class OpenCLChannelsDepthBufferProcessorImpl : OpenCLDepthBufferProcessorImpl
{
	virtual bool initProgram();
	virtual void run(const unsigned char* buffer, int length);
	virtual bool readProgram(std::string &source) const;

	size_t buf_channels_size, buf_gaussian_kernel_size;

	cl::Kernel kernel_processPixelStage2_phase_channels;
	cl::Kernel kernel_processPixelStage2_phase_channels3;
  cl::Kernel kernel_processPixelStage2_depth_channels;
	cl::Kernel kernel_processPixelStage2_depth_channels3;
	cl::Kernel kernel_filter_channels;
	cl::Kernel kernel_filter_channels2;

	cl::Buffer buf_phase_1;
	cl::Buffer buf_phase_2;
	cl::Buffer buf_phase_3;
  cl::Buffer buf_channels_1;
	cl::Buffer buf_channels_2;
  cl::Buffer buf_channels_3;
	cl::Buffer buf_channels_4;
	cl::Buffer buf_channels_1_filtered;
	cl::Buffer buf_channels_2_filtered;
	cl::Buffer buf_channels_1_phase_1;
	cl::Buffer buf_channels_2_phase_1;
	cl::Buffer buf_channels_3_phase_1;
	cl::Buffer buf_channels_4_phase_1;
	cl::Buffer buf_channels_1_phase_2;
	cl::Buffer buf_channels_2_phase_2;
	cl::Buffer buf_channels_3_phase_2;
	cl::Buffer buf_channels_4_phase_2;
};

class OpenCLShortChannelsDepthBufferProcessorImpl : OpenCLDepthBufferProcessorImpl
{
	virtual bool initProgram();
	virtual void run(const unsigned char* buffer, int length);
	virtual bool readProgram(std::string &source) const;

	size_t buf_gaussian_kernel_size buf_shortchannels_size;

	cl::Kernel kernel_processPixelStage2_phase_channels;
	cl::Kernel kernel_processPixelStage2_phase_channels3;
  cl::Kernel kernel_processPixelStage2_depth_channels;
	cl::Kernel kernel_processPixelStage2_depth_channels3;
	cl::Kernel kernel_filter_channels;
	cl::Kernel kernel_filter_channels2;

	cl::Buffer buf_phase_1;
	cl::Buffer buf_phase_2;
	cl::Buffer buf_phase_3;
  cl::Buffer buf_channels_1;
	cl::Buffer buf_channels_2;
  cl::Buffer buf_channels_3;
	cl::Buffer buf_channels_4;
	cl::Buffer buf_channels_1_filtered;
	cl::Buffer buf_channels_2_filtered;
	cl::Buffer buf_channels_1_phase_1;
	cl::Buffer buf_channels_2_phase_1;
	cl::Buffer buf_channels_3_phase_1;
	cl::Buffer buf_channels_4_phase_1;
	cl::Buffer buf_channels_1_phase_2;
	cl::Buffer buf_channels_2_phase_2;
	cl::Buffer buf_channels_3_phase_2;
	cl::Buffer buf_channels_4_phase_2;
};

class OpenCLKdeDepthBufferProcessorImpl : OpenCLDepthBufferProcessorImpl
{
	virtual bool initProgram();
	virtual void run(const unsigned char* buffer, int length);
	virtual bool readProgram(std::string &source) const;

	size_t buf_gaussian_kernel_size;

	cl::Kernel kernel_filter_kde;

	cl::Buffer buf_phase_1;
	cl::Buffer buf_phase_2;
	cl::Buffer buf_phase_3;
	cl::Buffer buf_conf1;
	cl::Buffer buf_conf2;
	cl::Buffer buf_conf3;
	cl::Buffer buf_kde_1;
	cl::Buffer buf_kde_2;
};

}
