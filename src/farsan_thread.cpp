#include "farsan_thread.h"


/*auto initSleep()
{
#ifdef WIN32
	//windows
#else
	auto start_time = std::chrono::steady_clock::now();
	return (start_time + frame_duration(1));
#endif
}

void sleepUntil(auto time)
{
#ifdef WIN32
	//windows
#else
	std::this_thread::sleep_until(time);
#endif
}*/

FarsanThread::FarsanThread()
{

}

FarsanThread::~FarsanThread()
{

}

void FarsanThread::spawn(THREAD_RET_VAL (*aFunction)(THREAD_ARG_TYPE), void * aArg)
{
#ifdef _WIN32
    mThread = CreateThread(NULL,0,aFunction,(LPVOID)aArg,0,&mThreadID);

#else
#if __cplusplus > 199711L

	mThread = std::thread(aFunction,aArg);

#else
	int ret = pthread_create(&mThread, NULL, aFunction, aArg);
#endif

#endif
}

void FarsanThread::join()
{
#ifdef WIN32
	 CloseHandle(mThread);
#else
#if __cplusplus > 199711L
	mThread.join();
#else
	pthread_join(mThread, NULL);
#endif
#endif
}

FarsanLock::FarsanLock()
{
#ifdef WIN32
  mMutex = CreateMutex( NULL, FALSE, NULL);
#else
#if __cplusplus > 199711L

#else
	pthread_mutex_init(&mMutex, NULL);// = PTHREAD_MUTEX_INITIALIZER;
#endif
#endif
}

FarsanLock::~FarsanLock()
{

}

FARSAN_LOCK_RETURN FarsanLock::lock()
{
#ifdef WIN32
	DWORD res = WaitForSingleObject(mMutex, INFINITE);
    if(res == WAIT_ABANDONED ||res == WAIT_FAILED){
        printf("Mutex lock failed...");
        exit(-1);

    }
#else
#if __cplusplus > 199711L
	return std::unique_lock<std::mutex> (mMutex);
#else
	pthread_mutex_lock( &mMutex );
#endif
#endif
}

void FarsanLock::unlock(FARSAN_UNLOCK_ARG)
{
#ifdef WIN32
   if (! ReleaseMutex(mMutex))
    {
       printf("MUTEX REALSE FAILED\n");
    }
#else
#if __cplusplus > 199711L
	lock.unlock();
#else
	pthread_mutex_unlock( &mMutex );
#endif
#endif
}

FARSAN_GET_MUTEX_RETURN FarsanLock::getMutex()
{
	return &mMutex;
}

FarsanCondVar::FarsanCondVar()
{
#ifdef WIN32
//  InitializeConditionVariable(&mCv);
//  InitializeCriticalSection (&mBufferLock);
#else
#if __cplusplus > 199711L

#else
	pthread_cond_init(&mCv, NULL);//   = PTHREAD_COND_INITIALIZER;
#endif
#endif
}

FarsanCondVar::~FarsanCondVar()
{

}

void FarsanCondVar::wait(COND_WAIT_ARG)
{
#ifdef WIN32
    lock->unlock();
	usleep(500);//Sleep(5);
	lock->lock();
#else
#if __cplusplus > 199711L
	mCv.wait(lock);
#else
	pthread_cond_wait( &mCv, lock->getMutex() );
#endif
#endif
}

void FarsanCondVar::notifyOne()
{
#ifdef WIN32
   // WakeConditionVariable (&mBufferLock);
#else
#if __cplusplus > 199711L
	mCv.notify_one();
#else
	pthread_cond_signal(&mCv);
#endif
#endif
}

void FarsanCondVar::notifyAll()
{
#ifdef WIN32
	//windows
#else
#if __cplusplus > 199711L
	mCv.notify_all();
#else
	pthread_cond_broadcast(&mCv);
#endif
#endif
}
