
#ifndef FARSAN_THREAD_H
#define FARSAN_THREAD_H



#ifdef _WIN32
#define _WIN32_WINNT 0x0900
#include <windows.h>
#include <winbase.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#define FARSAN_LOCK_RETURN void
#define FARSAN_UNLOCK_ARG void
#define COND_WAIT_ARG FarsanLock* lock
#define FARSAN_GET_MUTEX_RETURN HANDLE*
#define THREAD_RET_VAL DWORD WINAPI
#define THREAD_ARG_TYPE LPVOID

#else

#if __cplusplus > 199711L
#include <thread>
#include <mutex>
#include <condition_variable>
#define FARSAN_LOCK_RETURN std::unique_lock<std::mutex>
#define FARSAN_UNLOCK_ARG std::unique_lock<std::mutex>& lock
#define FARSAN_GET_MUTEX_RETURN std::mutex*
#define COND_WAIT_ARG std::unique_lock<std::mutex>& lock
#define THREAD_RET_VAL void*
#define THREAD_ARG_TYPE void*

#else
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#define FARSAN_LOCK_RETURN void
#define FARSAN_UNLOCK_ARG void
#define FARSAN_GET_MUTEX_RETURN pthread_mutex_t*
#define COND_WAIT_ARG FarsanLock* lock
#define THREAD_RET_VAL void*
#define THREAD_ARG_TYPE void*
#endif

#endif

class FarsanThread {

public:
	FarsanThread();
	~FarsanThread();
  void spawn(THREAD_RET_VAL (*aFunction)(THREAD_ARG_TYPE), void * aArg);
	void join();
private:
#ifdef _WIN32
	HANDLE mThread;
	DWORD mThreadID;
#else 
	#if __cplusplus > 199711L
		std::thread mThread;
	#else
		pthread_t mThread;
	#endif
#endif

};


class FarsanLock {

public:
	FarsanLock();
	~FarsanLock();
	FARSAN_LOCK_RETURN lock();
	void unlock(FARSAN_UNLOCK_ARG);
	FARSAN_GET_MUTEX_RETURN getMutex();
private:
#ifdef _WIN32
	HANDLE mMutex;
#else
	#if __cplusplus > 199711L
		std::mutex mMutex;
	#else
		pthread_mutex_t mMutex;
	#endif
#endif
};

class FarsanCondVar {

public:
	FarsanCondVar();
	~FarsanCondVar();

	void wait(COND_WAIT_ARG);
	void notifyOne();
	void notifyAll();

private:
#ifdef _WIN32
    //CRITICAL_SECTION   mBufferLock;
	//CONDITION_VARIABLE  mCv;

#else
	#if __cplusplus > 199711L
		std::condition_variable mCv;
	#else
		pthread_cond_t mCv;
	#endif
#endif
};

#endif

