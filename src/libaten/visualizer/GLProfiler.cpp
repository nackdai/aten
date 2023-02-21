#include "GLProfiler.h"
#include "visualizer/atengl.h"

namespace aten
{
    uint32_t GLProfiler::s_query[2] = { 0 };
    bool GLProfiler::s_isProfiling = false;
    bool GLProfiler::s_enable = false;

    void GLProfiler::start()
    {
        if (s_query[0] == 0) {
            CALL_GL_API(::glGenQueries(2, s_query));
        }
    }

    void GLProfiler::terminate()
    {
        if (s_query[0] > 0) {
            CALL_GL_API(::glDeleteQueries(2, s_query));

            s_query[0] = 0;
            s_query[1] = 0;
        }
    }

    void GLProfiler::begin()
    {
        if (!s_enable) {
            return;
        }

        AT_ASSERT(s_query[0] > 0);
        AT_ASSERT(!s_isProfiling);

        if (!s_isProfiling) {
            CALL_GL_API(::glQueryCounter(s_query[0], GL_TIMESTAMP));
            s_isProfiling = true;
        }
    }

    double GLProfiler::end()
    {
        if (!s_enable) {
            return 0.0;
        }

        AT_ASSERT(s_query[1] > 0);
        AT_ASSERT(s_isProfiling);

        if (s_isProfiling) {
            CALL_GL_API(::glFlush());
            CALL_GL_API(::glFinish());

            CALL_GL_API(::glQueryCounter(s_query[1], GL_TIMESTAMP));

            // wait until the results are available
            int32_t stopTimerAvailable = 0;
            while (!stopTimerAvailable) {
                CALL_GL_API(::glGetQueryObjectiv(
                    s_query[1],
                    GL_QUERY_RESULT_AVAILABLE,
                    &stopTimerAvailable));
            }

            GLuint64 startTime, stopTime;

            // get query results.
            CALL_GL_API(::glGetQueryObjectui64v(s_query[0], GL_QUERY_RESULT, &startTime));
            CALL_GL_API(::glGetQueryObjectui64v(s_query[1], GL_QUERY_RESULT, &stopTime));

            s_isProfiling = false;

            double elapsed = (stopTime - startTime) / 1000000.0;

            return elapsed;
        }

        return 0.0;
    }

    void GLProfiler::enable()
    {
        s_enable = true;
    }

    void GLProfiler::disable()
    {
        s_enable = false;
    }

    void GLProfiler::trigger()
    {
        s_enable = !s_enable;
    }

    bool GLProfiler::isEnabled()
    {
        return s_enable;
    }
}
