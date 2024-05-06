#pragma once

#include <functional>

#include "types.h"

namespace aten
{
    /**
    * タイムライン
    */
    class Timeline {
    public:
        Timeline();
        Timeline(
            float duration,
            float delay);

        ~Timeline() {}

    public:
        /** Initialize timeline.
         */
        void init(
            float duration,
            float delay);

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
        void advance(float delta);

        /** Rewind timeline.
         */
        void rewind();

        using TimeOverHandler = std::function<void(const Timeline&)>;

        /** Set handler if timeline is over the specified time.
         */
        void setTimeOverHandler(TimeOverHandler handler);

        /** Get current time.
         */
        float getTime() const;

        /** Get duration.
         */
        float getDuration() const;

        /** Get normalized time.
         */
        float getNormalized() const;

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
        void overrideTimeForcibly(float t);

    private:
        void toggleDirection();

        void setDuration(float duration);

        float getOverTime();
        void setOverTime(float over);

    protected:
        float m_Time{ float(0) };        // 時間
        float m_DelayTime{ float(0) };

        float m_Duration{ float(0) };    // 期間
        float m_Delay{ float(0) };       // 遅延

        float m_OverTime{ float(0) };

        bool m_isLoop{ false };    // ループするか
        bool m_isReverse{ false };    // 逆回転するかどうか
        bool m_isPause{ false };    // ポーズ中かどうか
        bool m_isForward{ true };    // 順方向進行かどうか

        TimeOverHandler m_TimeOverHandler;
    };

}
