cmake_minimum_required(VERSION 2.6)
project (LOGKINECT)

find_package( OpenCV )
#find_package(turbojpeg)
#find_package(hdf5)
#find_package(hdf5_hl)
#find_package(pthread)

set (LOGKINECT_INCLUDE_DIR include)
set (LOGKINECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

include_directories(include)
include_directories(kinect_parameters)

set(logkinect_src
  ${LOGKINECT_SOURCE_DIR}/depth_buffer_processor.cpp
  ${LOGKINECT_SOURCE_DIR}/buffer_handler.cpp
	${LOGKINECT_SOURCE_DIR}/rgb_buffer_processor.cpp
	${LOGKINECT_SOURCE_DIR}/farsan_thread.cpp
	${LOGKINECT_SOURCE_DIR}/read_file_handler.cpp
)

set(logkinect_inc
  ${LOGKINECT_INCLUDE_DIR}/depth_buffer_processor.h
  ${LOGKINECT_INCLUDE_DIR}/buffer_handler.h
	${LOGKINECT_INCLUDE_DIR}/rgb_buffer_processor.h
	${LOGKINECT_INCLUDE_DIR}/farsan_thread.h
	${LOGKINECT_INCLUDE_DIR}/read_file_handler.h
)

add_executable(example ${logkinect_src} example.cpp)
add_library(LOGKINECT SHARED ${logkinect_src} ${logkinect_inc})
#add_executable(logkinect ${logkinect_src})
target_link_libraries(LOGKINECT pthread turbojpeg hdf5 hdf5_hl OpenCL)
target_link_libraries(example pthread turbojpeg hdf5 hdf5_hl OpenCL ${OpenCV_LIBS})
#target_include_directories(logkinect ${LOGKINECT_INCLUDE_DIR})
