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
    TP_ARGS(),
    TP_FIELDS()
)

TRACEPOINT_EVENT(
    HPX_ALG,
    benchmark_exe_stop,
    TP_ARGS(),
    TP_FIELDS()
)

#endif // _TRACEPOINTS_H

#include <lttng/tracepoint-event.h>
