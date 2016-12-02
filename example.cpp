/* 
 * Reads Kinect v2 data stream from .h5 file and displays processed data
 */

#include "depth_buffer_processor.h"
#include "rgb_buffer_processor.h"
#include "buffer_handler.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

void saveFloatArrayToFileBin(float* pts, int len, std::string name)
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


int main(int argc, char** argv)
{
	std::string filename;

	if(argc > 1)
	{
		filename = argv[1];
	}
	else
	{
		std::cout<<"need filename.. \n";
		return -1;
	}
	std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(3);	

	unsigned int num_images = 2000;
	int bilateral_filter = 1;
	int dev = -1;
	int pipeline = 10;//  = libfreenect2 outlier rejection, 1 = spatial propagation, 2 = libfreenect2 without outlier rejection, 3 = spatial propagation 3 hypothesis, 4 = output 2 hypothesis with weights, 5 = channels, 6 = channels3, 7 = enable more channels for 2 hypotheses, 8 = channels with half-floats, 9 channels with ushort, 10 = kde two hypotheses, 11 = kde three hypotheses
	logkinect::Depth_Packet  depth_packet_libfreenect_, ir_packet_libfreenect_;
	logkinect::Color_Packet color_packet_;
	logkinect::LogBufferHandler* mBufferHandler = new logkinect::LogBufferHandler(filename, 30,"/home/felix/Documents/SVN/kinect_v2/simulator/logkinect/kinect_parameters/cam_default_params_1.h5");
	logkinect::OpenCLDepthBufferProcessor* mDepthBufferProcessor_libfreenect = new logkinect::OpenCLDepthBufferProcessor(mBufferHandler->m_p0TableBuffer, mBufferHandler->m_p0TableLength, mBufferHandler->ir_intrinsics, mBufferHandler->rgb_intrinsics, mBufferHandler->rot, mBufferHandler->trans, pipeline, 0, bilateral_filter,dev);
	
	logkinect::RgbBufferProcessor mRgbBufferProcessor;
	
	mBufferHandler->start();
	mRgbBufferProcessor.start();
	unsigned char* ir_buffer, *rgb_buffer;
	int ir_buffer_length, rgb_buffer_length;
	unsigned int cnt = 0;

	while(mBufferHandler->running() && cnt < num_images)
	{
		mBufferHandler->readNextIrFrame(&ir_buffer, &ir_buffer_length);
		mBufferHandler->readNextRgbFrame(&rgb_buffer, &rgb_buffer_length);
		mDepthBufferProcessor_libfreenect->process(ir_buffer,ir_buffer_length);
		mRgbBufferProcessor.process(rgb_buffer,rgb_buffer_length);

		mDepthBufferProcessor_libfreenect->readResult(depth_packet_libfreenect_);
		if(dev > 0)
		{
			mDepthBufferProcessor_libfreenect->readIrResult(ir_packet_libfreenect_);
			saveFloatArrayToFileBin((float*)ir_packet_libfreenect_.buffer, 512*424*3, "bibliotek_phi_bilatoff.bin");
		}
		mRgbBufferProcessor.readResult(color_packet_);
		
		cv::Mat depth = cv::Mat(depth_packet_libfreenect_.height,depth_packet_libfreenect_.width,CV_32FC1,depth_packet_libfreenect_.buffer);
		//cv::Mat conf = cv::Mat(depth_packet_libfreenect_.height,depth_packet_libfreenect_.width,CV_32FC1,depth_packet_libfreenect_.buffer+512*424*4);
		cv::Mat color = cv::Mat(color_packet_.height,color_packet_.width,CV_8UC3,color_packet_.buffer);
		
		cv::cvtColor(color, color, CV_BGR2RGB);

		//cv::namedWindow( "Display conf", cv::WINDOW_AUTOSIZE );// Create a window for display.
		//cv::imshow( "Display color", 0.1f*conf);
		cv::namedWindow( "Display depth", cv::WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display depth", 2*depth/18750.0f);
		cv::namedWindow( "Display color", cv::WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display color", color);
		//cv::waitKey(3);
		char key = cv::waitKey(0);
		if(key == 's')
		{
			std::cout<<"save data..\n";
			std::stringstream ss, pipeline_ss;
			ss << cnt;
			pipeline_ss << pipeline;
			std::string color_name = "../../color/color_fysik";
			color_name.append(ss.str());
			imwrite(color_name.append(".png"),color,compression_params);
			
			std::string d_name = "depth_fysik_pipeline_";
			d_name.append(pipeline_ss.str());
			d_name.append("_frame_");
			d_name.append(ss.str());
			//imwrite(d_name.append(".png"),255*depth/18750.0f,compression_params);
			saveFloatArrayToFileBin((float*)depth_packet_libfreenect_.buffer, 512*424, d_name.append(".bin"));
		}
		std::cout<<cnt<<std::endl;
		//std::cout<<"key = "<<key<<std::endl;
		cnt++;
		delete[] depth_packet_libfreenect_.buffer;
	}
	
	delete mBufferHandler;
	delete mDepthBufferProcessor_libfreenect;
	return 0;
}
