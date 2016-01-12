/* 
 * Reads Kinect v2 data stream from .h5 file and displays processed data
 */

#include "depth_buffer_processor.h"
#include "rgb_buffer_processor.h"
#include "buffer_handler.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

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

	unsigned int num_images = 25;
	int pipeline =  0; //0 = libfreenect2 outlier rejection, 1 = spatial propagation, 2 = libfreenect2 without outlier rejection, 3 = spatial propagation 3 hypothesis, 4 = output 2 hypothesis with weights, 5 = channels, 6 = channels3, 7 = enable more channels for 2 hypotheses
	logkinect::Depth_Packet  depth_packet_libfreenect_;
	logkinect::Color_Packet color_packet_;
	logkinect::LogBufferHandler* mBufferHandler = new logkinect::LogBufferHandler(filename, 100);
	logkinect::OpenCLDepthBufferProcessor* mDepthBufferProcessor_libfreenect = new logkinect::OpenCLDepthBufferProcessor(mBufferHandler->m_p0TableBuffer, mBufferHandler->m_p0TableLength, mBufferHandler->ir_intrinsics, mBufferHandler->rgb_intrinsics, mBufferHandler->rot, mBufferHandler->trans, pipeline, 0,-1);
	
	logkinect::RgbBufferProcessor mRgbBufferProcessor;
	
	mBufferHandler->start();
	mRgbBufferProcessor.start();
	unsigned char* ir_buffer, *rgb_buffer;
	int ir_buffer_length, rgb_buffer_length;
	unsigned int cnt = 0;

	while(mBufferHandler->running() && cnt<num_images+10)
	{
		mBufferHandler->readNextIrFrame(&ir_buffer, &ir_buffer_length);
		mBufferHandler->readNextRgbFrame(&rgb_buffer, &rgb_buffer_length);
		mDepthBufferProcessor_libfreenect->process(ir_buffer,ir_buffer_length);
		mRgbBufferProcessor.process(rgb_buffer,rgb_buffer_length);

		mDepthBufferProcessor_libfreenect->readResult(depth_packet_libfreenect_);

		mRgbBufferProcessor.readResult(color_packet_);

		cv::Mat depth = cv::Mat(depth_packet_libfreenect_.height,depth_packet_libfreenect_.width,CV_32FC1,depth_packet_libfreenect_.buffer);
		cv::Mat color = cv::Mat(color_packet_.height,color_packet_.width,CV_8UC3,color_packet_.buffer);
		
		cv::namedWindow( "Display libfreenect", cv::WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display libfreenect", depth/18750.0f);
		cv::namedWindow( "Display libfreenect", cv::WINDOW_AUTOSIZE );// Create a window for display.
		cv::imshow( "Display libfreenect", color);
		cv::waitKey(3);
		cnt++;
		delete[] depth_packet_libfreenect_.buffer;
	}
	
	delete mBufferHandler;
	delete mDepthBufferProcessor_libfreenect;
	return 0;
}
