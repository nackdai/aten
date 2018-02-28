#include "misc/timeline.h"
#include "math/math.h"

namespace aten
{
	// コンストラクタ
	Timeline::Timeline()
	{
	}

	Timeline::Timeline(
		real duration,
		real delay)
	{
		m_Duration = duration;
		m_Delay = delay;
	}

	// 初期化
	void Timeline::init(
		real duration,
		real delay)
	{
		m_Duration = duration;
		m_Delay = delay;
	}

	void Timeline::start()
	{
		m_isPause = false;
	}

	// ストップ
	void Timeline::stop()
	{
		m_Time = (isForward() ? 0.0f : m_Duration);
		m_isPause = true;
	}

	void Timeline::pause()
	{
		m_isPause = true;
	}

	// リセット
	void Timeline::reset()
	{
		m_Time = real(0);
		m_DelayTime = real(0);
		m_OverTime = real(0);
		m_isForward = true;
	}

	// 進行
	void Timeline::advance(real delta)
	{
		if (isPaused()) {
			// ポーズ中なので何もしない
			return;
		}

		if (m_DelayTime < m_Delay) {
			// 遅延
			m_DelayTime += delta;

			if (m_DelayTime < m_Delay) {
				// まだまだ始まらない
				return;
			}

			delta = m_DelayTime - m_Delay;
		}

		if (m_OverTime != 0.0f) {
			if (isForward()) {
				m_Time += m_OverTime;
			}
			else {
				m_Time -= m_OverTime;
			}

			m_OverTime = 0.0f;
		}

		bool isOver = false;

		if (isForward()) {
			m_Time += delta;
			isOver = (m_Time >= m_Duration);
		}
		else {
			// 逆方向
			m_Time -= delta;
			isOver = (m_Time <= 0.0f);
		}

		if (isOver) {
			m_OverTime = (isForward() ? m_Time - m_Duration : aten::abs(m_Time));
			m_Time = (isForward() ? m_Duration : 0.0f);

			if (isLoop() && willReverse()) {
				// 向きを変える
				rewind();
			}

			pause();

			if (isLoop()) {
				stop();

				// ループするので再開
				start();
			}

			if (m_TimeOverHandler) {
				m_TimeOverHandler(*this);
			}
		}
	}

	void Timeline::rewind()
	{
		m_isForward = !m_isForward;
	}

	void Timeline::setTimeOverHandler(TimeOverHandler handler)
	{
		m_TimeOverHandler = handler;
	}

	real Timeline::getTime() const
	{
		return m_Time;
	}

	real Timeline::getDuration() const
	{
		return m_Duration;
	}

	real Timeline::getNormalized() const
	{
		real ret = (m_Duration != 0.0f ? m_Time / m_Duration : 0.0f);
		return ret;
	}

	void Timeline::enableLoop(bool enable)
	{
		m_isLoop = enable;
	}

	bool Timeline::isLoop() const
	{
		return m_isLoop;
	}

	void Timeline::autoReverse(bool enable)
	{
		m_isReverse = enable;
	}

	bool Timeline::willReverse() const
	{
		return m_isReverse;
	}

	bool Timeline::isPaused() const
	{
		return m_isPause;
	}

	bool Timeline::isForward() const
	{
		return m_isForward;
	}

	// Override current time forcibly.
	void Timeline::overrideTimeForcibly(real t)
	{
		m_Time = t;
		m_OverTime = 0.0f;
	}

	void Timeline::toggleDirection()
	{
		m_isForward = !m_isForward;
	}

	void Timeline::setDuration(real duration)
	{
		m_Duration = duration;
	}

	real Timeline::getOverTime()
	{
		return m_OverTime;
	}

	void Timeline::setOverTime(real over)
	{
		m_OverTime = over;
	}
}
