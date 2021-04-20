#include "defs.h"
#include "misc/timer.h"

namespace AT_NAME {
    void timer::init()
    {
    }

    void timer::begin()
    {
        gettimeofday(&m_begin, NULL);
    }

    real timer::end()
    {
        timeval end;
        gettimeofday(&end, NULL);

        long sec = end.tv_sec - m_begin.tv_sec;
        long usec = end.tv_usec - m_begin.tv_usec;
        real msec = sec * real(1000.0) + (real)usec / real(1000.0);
        return msec;
    }

    SystemTime timer::getSystemTime()
    {
        // NOTE
        // http://www9.plala.or.jp/sgwr-t/lib/localtime.html
        // http://labo.utsubo.tokyo/2014/03/06/linux%E3%81%A7%E3%83%9F%E3%83%AA%E7%A7%92%E3%81%BE%E3%81%A7%E5%8F%96%E5%BE%97/

        timeval tv;
        gettimeofday(&tv, NULL);

        auto tmp = localtime(&tv.tv_sec);

        SystemTime ret;
        ret.year = tmp->tm_year + 1900;
        ret.month = tmp->tm_mon + 1;
        ret.dayOfWeek = tmp->tm_wday;
        ret.day = tmp->tm_mday;
        ret.hour = tmp->tm_hour;
        ret.minute = tmp->tm_min;
        ret.second = tmp->tm_sec;
        ret.milliSeconds = tv.tv_usec / 1000;

        return ret;
    }
}
