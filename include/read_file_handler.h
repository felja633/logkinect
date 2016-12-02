#include "H5Cpp.h"
#include <iostream>
#include <string>
#include <vector>

#ifndef READ_FILE_HANDLER_H
#define READ_FILE_HANDLER_H

class ReadFileHandler {

public:
  ReadFileHandler(const std::string filename);
  ~ReadFileHandler();

  void ReadRgbBuffer(unsigned char** arr, int* length, int frame);
	void ReadRgbBuffer(unsigned char** arr, int* length, int** timestamp, int frame);
  void ReadIrBuffer(unsigned char** arr, int* length, int frame);
	void ReadIrBuffer(unsigned char** arr, int* length, int** timestamp, int frame);
  void ReadBuffer(unsigned char** arr, int* length, std::string data_loc);  
  void ReadFrameHostTimeStamp(uint64_t** timestamp, int frame_num);
	int number_of_groups;
private:
  hid_t getGroup(int frame_num);
  void ReadCharArray(unsigned char** arr, int* length, hid_t identifier, std::string dataset);
	void ReadIntArray(int** arr, int* length, hid_t identifier, std::string dataset);
	void ReadUInt64Array(uint64_t** arr, int* length, hid_t identifier, std::string dataset);
  hid_t mFileId;
};



#endif

