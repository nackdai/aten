#pragma once

#include <thread>
#include <atomic>

#include "defs.h"
#include "types.h"

namespace aten
{
    /**
     */
    class Thread {
    public:
        /** 実行中のスレッドを、指定されたミリ秒数の間、スリープ.
         */
        static void sleep(uint32_t millisec);

        /** 現在実行中のスレッドを一時的に休止させ、ほかのスレッドが実行できるようにする.
         *
         * 同じプロセッサ上で動いているスレッド間でのみ可能
         *
         * NOTE
         * windowsだとwinbase.hの中で#define Yield()としているため、関数名にYieldが使えない
         */
        static void yieldThread();

    public:
        Thread();
        Thread(const char* name);

        virtual ~Thread();

    public:
        /** このスレッドの実行を開始.
         */
        virtual bool start(
            std::function<void(void*)> func,
            void* userData);

        /** このスレッドの実行を開始.
         */
        virtual bool start();

        /** スレッド実行中かどうかを取得.
         */
        bool isRunning();

		virtual void run();

        /** このスレッドが終了するのを待機.
         */
		virtual void join();

        /** 名前取得.
         */
        const char* getName() const;

    protected:
        uint32_t m_ThreadResult;

		std::thread m_thread;

        std::string m_Name;

        bool m_isRunning{ false };

        std::function<void(void*)> m_func{ nullptr };
        void* m_userData{ nullptr };
    };
}
