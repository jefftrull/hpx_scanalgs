#!/usr/bin/python3
# Script for analyzing tracepoint data from HPX exclusive scan execution

# Copyright 2019 Jeff Trull <edaskel@att.net>
# MIT license

# This runs on lttng traces with the following user events defined:
# benchmark_exe_start/stop : marks the start and end of a single algorithm run
# chunk_start/stop: records duration and other info for a chunk

# and the following context info (presently unused):
# perf:thread:cpu-cycles  (cycle counter)
# perf:thread:LLC-load-misses (L3 load misses)

# so
# start the target executable paused (if it can be paused)
# for example:
# ./exsvp --benchmark_filter=Parallel/16777216/real_time
# then in another shell:
# lttng create your-session-name
# lttng enable-event --userspace HPX:'*'
# lttng enable-event --userspace HPX_ALG:'*'
# lttng add-context -u --type=perf:thread:cpu-cycles
# lttng start
# resume your target executable/benchmark (hit return, for exsvp)
# then when it finishes:
# lttng stop
# now run this script:
# analyze_trace.py ~/lttng-traces/your-session-name-whatever

# You need to use the feature/parallel_alg_tracepoints branch
# from the https://github.com/jefftrull/hpx fork
# then build e.g. the exsvp target in this project

import sys
import operator
from itertools import takewhile, islice, dropwhile
from functools import reduce

import babeltrace   # version 1.0
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

def display_bm(title, evts, seq_runtime=None):
    """Display the benchmark starting at the current offset in the iterator"""

    # just get the first one, for now
    starttimes = {}   # key is (start_ofs, stop_ofs, stage)
    f1_durations = {} # key is (start_ofs, stop_ofs)
    f3_durations = {} # key is (start_ofs, stop_ofs)

    basetime = None
    chunksize = -1
    inpsize = -1
    for event in takewhile(lambda evt: evt.name != 'HPX_ALG:benchmark_exe_stop',
                           evts):
        if (event.name == 'HPX:chunk_start' or event.name == 'HPX:chunk_stop') and event['stage'] == 2:
            continue
        if event.name == 'HPX:chunk_start':
            if basetime == None:
                basetime = event.timestamp

            starttimes[(event['start_ofs'], event['stop_ofs'], event['stage'])] = event.timestamp - basetime
            # collect some helpful info for displaying the Y axis
            if event['start_ofs'] == 0:
                chunksize = event['stop_ofs']
            if event['stop_ofs'] > inpsize:
                inpsize = event['stop_ofs']

        if event.name == 'HPX:chunk_stop':
            start = starttimes[(event['start_ofs'], event['stop_ofs'], event['stage'])]
            stop = event.timestamp - basetime
            if event['stage'] == 1:
                f1_durations[(event['start_ofs'], event['stop_ofs'])] = (start, stop)
            else:
                f3_durations[(event['start_ofs'], event['stop_ofs'])] = (start, stop)

    fig, ax = plt.subplots()

    for chunk in f1_durations:
        ax.broken_barh([(f1_durations[chunk][0], (f1_durations[chunk][1] - f1_durations[chunk][0])),
                        (f3_durations[chunk][0], (f3_durations[chunk][1] - f3_durations[chunk][0]))],
                       (chunk[0], chunk[1] - chunk[0]),
                       facecolors=('tab:red', 'tab:orange'))
    # use chunk boundaries for y axis ticks
    assert(chunksize != -1)
    assert(inpsize != -1)
    ax.set_yticks(range(0, inpsize+1, chunksize))
    ax.grid(True)

    ax.set_xlabel('time (ns)')
    ax.set_ylabel('chunk extent')
    ax.set_title(title)
    plt.gca().invert_yaxis()

    # also mark the sequential alg's runtime for comparison
    seqline = None
    if seq_runtime:
        seqline = plt.axvline(x=seq_runtime, linestyle='--', color='gray', label='Seq Runtime')

    # a helpful legend
    handles = [mpatches.Patch(color='red', label='Stage 1'),
                mpatches.Patch(color='orange', label='Stage 3')]
    if seq_runtime:
        handles.append(seqline)
    plt.legend(handles=handles)

    plt.show()


col = babeltrace.TraceCollection()
tracedict = col.add_traces_recursive(sys.argv[1], 'ctf')
if tracedict is None:
    raise RuntimeError('Cannot add trace')

   
#
# Second project: see if (temporal) distance between stages affects execution time
#

# this is basically a calculation of the correlation between cpu cycles as measured by
# the performance counter vs. aggregate distance between stages for all chunks
# we could follow this up with an L3 miss correlation check using that perf counter

bm_start_ts = []
bm_runtimes = []
bm_stage_gaps = []

evts = col.events
for evt in evts:
    bm_start = evt.timestamp

    # handle benchmark start
    f1_stop     = {} # key is (start_ofs, stop_ofs)
    f3_start    = {} # key is (start_ofs, stop_ofs)
    for event in takewhile(lambda evt: evt.name != 'HPX_ALG:benchmark_exe_stop',
                          evts):
        # handle chunk start or stop
        if event.name == 'HPX:chunk_stop' and event['stage'] == 1:
            f1_stop[(event['start_ofs'], event['stop_ofs'])] = event.timestamp
        if event.name == 'HPX:chunk_start' and event['stage'] == 3:
            f3_start[(event['start_ofs'], event['stop_ofs'])] = event.timestamp

    # handle benchmark end
    bm_stop = evt.timestamp

    # now record three pieces of data:

    # 1) benchmark runtime
    assert(bm_stop > bm_start)
    bm_runtimes.append(bm_stop - bm_start)

    # 2) aggregate gaps between stage 1 and 3 summed over all chunks
    stage_gap_sum = reduce(lambda a, b: a + (f3_start[b] - f1_stop[b]),
                           f1_stop.keys(),
                           0)
    bm_stage_gaps.append(stage_gap_sum)

    # 3) the benchmark start time (to use as a key later)
    bm_start_ts.append(bm_start)

print('correlation of runtimes to stage gaps is %f' % pearsonr(bm_runtimes, bm_stage_gaps)[0])

# next project: display the shortest and longest runtimes as broken bar graphs like above
# to help me try to figure out what's happening
# can I store an iterator copy?
# NO, not reliably anyway
# instead use stored timestamp of the benchmark starts as a "key", then go back for each

# find ts of fastest benchmark run
fast_idx = min(enumerate(bm_runtimes), key=operator.itemgetter(1))[0]
fast_ts = bm_start_ts[fast_idx]
# and slowest (skipping the first, which is unusually slow)
# On IRC heller says "Stack allocation and associated page faults" is the reason
slow_idx = max(
    islice(enumerate(bm_runtimes), 1, None),
    key=operator.itemgetter(1))[0]
slow_ts = bm_start_ts[slow_idx]
print('fastest runtime was %d at timestamp %d (run %d)' % (bm_runtimes[fast_idx], fast_ts, fast_idx))
print('slowest runtime was %d at timestamp %d (run %d)' % (bm_runtimes[slow_idx], slow_ts, slow_idx))

# display a histogram of runtimes
fig, ax = plt.subplots()
ax.hist(list(islice(bm_runtimes, 1, None)), bins=50)
ax.set_title('Runtime Distribution')
ax.set_xlabel('ns')
plt.show()


# display slowest run
# NOTE: for the moment the sequential runtime is manually entered so this could be misleading
display_bm('Slowest Run: %dns' % bm_runtimes[slow_idx],
           dropwhile(lambda evt: evt.timestamp != slow_ts, col.events),
           )
#           9422885)
#           301670)
display_bm('Fastest Run: %dns' % bm_runtimes[fast_idx],
           dropwhile(lambda evt: evt.timestamp != fast_ts, col.events),
           )
#           9422885)
#           301670)

