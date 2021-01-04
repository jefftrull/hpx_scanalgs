// Tracepoint definitions for algorithm analysis
// This only works on suitably configured Linux systems...
// Author: Jeff Trull <edaskel@att.net>

#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER HPX_ALG

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./tracepoints.h"

#if !defined(_TRACEPOINTS_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _TRACEPOINTS_H

#include <lttng/tracepoint.h>

// benchmark single execution start/end
TRACEPOINT_EVENT(
    HPX_ALG,
    benchmark_exe_start,
    TP_ARGS(
        std::size_t, worker_thread
    ),
    TP_FIELDS(
        ctf_integer(std::size_t, worker_thread, worker_thread)
    )
)

TRACEPOINT_EVENT(
    HPX_ALG,
    benchmark_exe_stop,
    TP_ARGS(),
    TP_FIELDS()
)

// events just for local use (named similarly, but not the HPX events)
TRACEPOINT_EVENT(
    HPX_ALG,
    chunk_start,
    TP_ARGS(
        std::size_t, start_ofs,
        std::size_t, stop_ofs,
        int, stage
    ),
    TP_FIELDS(
        ctf_integer(std::size_t, start_ofs, start_ofs)
        ctf_integer(std::size_t, stop_ofs, stop_ofs)
        ctf_integer(int, stage, stage)
    )
)

TRACEPOINT_EVENT(
    HPX_ALG,
    chunk_stop,
    TP_ARGS(
        std::size_t, start_ofs,
        std::size_t, stop_ofs,
        int, stage
    ),
    TP_FIELDS(
        ctf_integer(std::size_t, start_ofs, start_ofs)
        ctf_integer(std::size_t, stop_ofs, stop_ofs)
        ctf_integer(int, stage, stage)
    )
)

// finally a single event for stage 2, which generally does just one simple op
// so we omit the stage number and data range
TRACEPOINT_EVENT(
    HPX_ALG,
    stage2,
    TP_ARGS(
        std::size_t, loc
    ),
    TP_FIELDS(
        ctf_integer(std::size_t, loc, loc)
    )
)

#endif // _TRACEPOINTS_H

#include <lttng/tracepoint-event.h>
