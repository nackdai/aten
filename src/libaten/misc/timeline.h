#pragma once

#include "types.h"

#include <functional>

namespace aten
{
    /**
    * タイムライン
    */
    class Timeline {
    public:
        Timeline();
        Timeline(
            real duration,
            real delay);

        ~Timeline() {}

    public:
        /** Initialize timeline. 
         */
        void init(
            real duration,
            real delay);

        /** Start timeline.
         */
        void start();

        /** Stop timeline.
         */
        void stop();

        /** Pause timeline
         */
        void pause();

        /** Reset timeline.
         */
        void reset();

        /** Advance timeline by specified delta time.
         */
        void advance(real delta);

        /** Rewind timeline.
         */
        void rewind();

        using TimeOverHandler = std::function<void(const Timeline&)>;

        /** Set handler if timeline is over the specified time.
         */
        void setTimeOverHandler(TimeOverHandler handler);

        /** Get current time.
         */
        real getTime() const;

        /** Get duration.
         */
        real getDuration() const;

        /** Get normalized time.
         */
        real getNormalized() const;

        /** Set if timeline is loop.
         */
        void enableLoop(bool enable);

        /** Get if timeline is loop.
         */
        bool isLoop() const;

        /** Set whether timeline is reverse when timeline is loop.
         */
        void autoReverse(bool enable);

        /** Get whether timeline is reverse when timeline is loop.
         */ 
        bool willReverse() const;

        /** Get whether timeline is posed.
         */
        bool isPaused() const;

        /** Get whether timeline runs forward.
         */
        bool isForward() const;

        /** Override current time forcibly.
         */
        void overrideTimeForcibly(real t);

    private:
        void toggleDirection();

        void setDuration(real duration);

        real getOverTime();
        void setOverTime(real over);

    protected:
        real m_Time{ real(0) };        // 時間
        real m_DelayTime{ real(0) };

        real m_Duration{ real(0) };    // 期間
        real m_Delay{ real(0) };       // 遅延

        real m_OverTime{ real(0) };

        bool m_isLoop{ false };    // ループするか
        bool m_isReverse{ false };    // 逆回転するかどうか
        bool m_isPause{ false };    // ポーズ中かどうか
        bool m_isForward{ true };    // 順方向進行かどうか

        TimeOverHandler m_TimeOverHandler;
    };

}
