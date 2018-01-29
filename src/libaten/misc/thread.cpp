#include "misc/thread.h"

namespace aten
{
	// 実行中のスレッドを、指定されたミリ秒数の間、スリープ.
	void Thread::sleep(uint32_t millisec)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
	}

	// 現在実行中のスレッドを一時的に休止させ、ほかのスレッドが実行できるようにする.
	void Thread::yieldThread()
	{
		std::this_thread::yield();
	}

	Thread::Thread()
	{
		m_ThreadResult = 0;
	}

	Thread::Thread(const char* name)
		: Thread()
	{
		m_Name = name;
	}

	Thread::~Thread()
	{
		join();
	}

	// このスレッドの実行を開始.
	bool Thread::start(
		std::function<void(void*)> func,
		void* userData)
	{
		m_isRunning = true;
		m_func = func;
		m_userData = userData;
		m_thread = std::thread([this] { run(); });

		return true;
	}

	// このスレッドの実行を開始.
	bool Thread::start()
	{
		m_isRunning = true;
		m_thread = std::thread([this] { run(); });

		return true;
	}

	// スレッド実行中かどうかを取得.
	bool Thread::isRunning()
	{
		return m_isRunning;
	}

	void Thread::run()
	{
		if (m_func) {
			m_func(m_userData);
		}
	}

	// このスレッドが終了するのを待機.
	void Thread::join()
	{
		if (m_isRunning) {
			m_thread.join();
			m_isRunning = false;
			m_userData = nullptr;
		}
	}

	// 名前取得.
	const char* Thread::getName() const
	{
		return m_Name.c_str();
	}
}
