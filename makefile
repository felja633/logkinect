CC=gcc
#CC=g++
#CFLAGS+=-g
#CCFLAGS = -g -pedantic -Wall -Wextra
#CFLAGS+= -O3 -Ofast -mtune=native -march=native
#CXXFLAGS= -O3 -Ofast -mtune=native -march=native
# -O3 -Ofast -mtune=native -march=native
#CFLAGS+=  `pkg-config --cflags opencv` -Wall -fpermissive  -lm 
#LDFLAGS+=  -lm -lstdc++ -lhdf5 -lhdf5_hl -lturbojpeg 
CLFLAGS= -lOpenCL 
#GLFLAGS= -lGL -lglut -lX11

#CPPFLAGS= -g -O3 -Ofast -mtune=native -march=native -std=c++0x -lGL -DGL_GLEXT_PROTOTYPES -Wno-write-strings
CPPFLAGS= -fPIC -shared -g -Wno-write-strings 
CFLAGS= -fPIC -shared -g -std=c99 -Wno-write-strings
LINK= -shared-libgcc -lstdc++ -lm -lhdf5 -lhdf5_hl -lpthread
# -lGLU

PIPELINE = 0
UNDISTORT = 0
CHANNELS = 0
NUM_CHANNELS = 32
CHANNEL_FILT_SIZE = 5

#CPPFLAGS+= -Wall -fpermissive -Iinclude -Ikinect_parameters -DPIPELINE=$(PIPELINE) -DUNDISTORT=$(UNDISTORT) -DCHANNELS=$(CHANNELS) -DNUM_CHANNELS=$(NUM_CHANNELS) -DCHANNEL_FILT_SIZE=$(CHANNEL_FILT_SIZE)

CPPFLAGS+= -Wall -fpermissive -Iinclude -Ikinect_parameters
LIBTARGET=lib/liblogkinect.so.1

VPATH = src:src 
OBJS= farsan_thread.o buffer_handler.o read_file_handler.o depth_buffer_processor.o 

#SRCFILES= farsan_thread.cpp farsan_dispatcher.cpp GL_utilities.c loadobj.c LoadTGA.c VectorUtils3.c buffer_handler.cpp read_file_handler.cpp depth_buffer_processor.cpp rgb_buffer_processor.cpp cl_helper.cpp cl_program.cpp cl_kernel.cpp GLHelper.cpp Renderable.cpp user_camera.cpp ar_handler.cpp ar_instance_base.cpp ar_smoke.cpp debug_mesh.cpp

.PHONY: all clean


$(LIBTARGET): $(OBJS)
	$(CC) $(CPPFLAGS) -Wl,-soname,liblogkinect.so.1 $(LINK) $(CLFLAGS) -o $@ $^ -lc 

#libsfarsan.a: $(OBJS)
#	ar crvs $@ $^

#lib/libsfarsan.so: $(SRCFILES)
#	$(CC) -fPIC -c $(CPPFLAGS) -o $(OBJS) $(LINK) $(CLFLAGS) 
#	$(CC) -shared  -Wl,-soname,libsfarsan.so.1 -o libsfarsan.so.1.0 $(OBJS) $(LINK) $(CLFLAGS) 
#	mv libsfarsan.so.1.0 lib
#	ln -sf lib/libsfarsan.so.1.0 lib/libsfarsan.so.1
#	ln -sf lib/libsfarsan.so.1.0 lib/libsfarsan.so

%.o: %.cpp
	$(CC) -c $(CPPFLAGS) $< 

%.o: %.c
	$(CC) -c $(CFLAGS) $< 

all: $(LIBTARGET)

clean:
	rm -f $(LIBTARGET) *.o 
