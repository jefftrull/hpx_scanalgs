// Parallel Exclusive Scan testbed
// a prototype done "manually" (using C++17 features) only

#include <vector>
#include <numeric>
#include <algorithm>
#include <future>
#include <iostream>
#include <iterator>
#include <cassert>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdio>

#include <emmintrin.h>

#define TRACEPOINT_DEFINE
#include "tracepoints.h"

#include <benchmark/benchmark.h>

void logtime(std::string const& log)
{
    std::cout << log << " at ";

    using namespace std::chrono;
    std::cout << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() << "\n";
}

template<typename FwdIter1, typename FwdIter2, typename T = typename FwdIter1::value_type, typename Op = std::plus<T>>
T sequential_exclusive_scan(FwdIter1 first, FwdIter1 last, FwdIter2 dest, T init = T(), Op op = Op())
{
    // taken from HPX, for maximum fidelity
    // appears to be about 10% faster than my approach
    T temp = init;
    for (; first != last; (void) ++first, ++dest)
    {
        init = std::invoke(op, init, *first);
        *dest = temp;
        temp = init;
    }
    return temp;

}

template<typename FwdIter1, typename FwdIter2, typename T = typename FwdIter1::value_type, typename Op = std::plus<T>>
T sequential_exclusive_scan_n(FwdIter1 first, std::size_t count, FwdIter2 dest, T init = T(), Op op = Op())
{
    // HPX code
    T temp = init;
    for (; count-- != 0; (void) ++first, ++dest)
    {
        init = std::invoke(op, init, *first);
        *dest = temp;
        temp = init;
    }
    return temp;
}

//
// sideband info for benchmarking/tracing
//

// performance tuning
std::size_t chunksize;
std::size_t thread_count = 4;

// where each chunk begins
std::vector<int>::iterator true_start;

template<typename T, typename FwdIter1, typename FwdIter2, typename Op = std::plus<T>>
std::pair<std::future<T>, std::vector<std::future<void>>>
exclusive_scan_mt_impl(FwdIter1 start, FwdIter1 end, FwdIter2 dst, T init, Op op,
                       std::future<T> phase2_input_handle,
                       std::vector<std::future<void>> completion_handles)
{
    std::size_t sz = std::distance(start, end);
    std::size_t partition_size = sz / thread_count;

    // launch appropriate tasks to do the next chunk of computation
    // each tasks waits to start until the analogous task in the previous chunk completes
    // the second phase of the tasks are chained together through the cumulative sum

    std::vector<std::future<void>> task_complete_handles;
    task_complete_handles.reserve(thread_count);

    FwdIter1 first = start;
    FwdIter2 ldst = dst;

    for (std::size_t i = 0;
         i < thread_count - 1;
         i++, std::advance(first, partition_size), std::advance(ldst, partition_size))
    {
        FwdIter1 last = std::next(first, partition_size);
        FwdIter2 llast = std::next(ldst, partition_size);

        std::promise<T> phase2_result_promise;
        std::future<T> phase2_result_handle = phase2_result_promise.get_future();

        task_complete_handles.push_back(
            std::async(std::launch::async,
                       [=,
                        prior_completion_handle = std::move(completion_handles[i]),
                        phase2_input_handle = std::move(phase2_input_handle),
                        phase2_result_promise = std::move(phase2_result_promise)
                        ]() mutable {
                           prior_completion_handle.get();
                           // phase 1: parallel exclusive scans on each partition
                           tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, last), 1);
                           T local_result = sequential_exclusive_scan(first, last, ldst);
                           tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, last), 1);
                           // store the accumulated result for the next partition
                           T prior_result = phase2_input_handle.get();
                           phase2_result_promise.set_value(op(prior_result, local_result));
                           // phase 2: update sequential scan using results from partitions 0..i-1
                           tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, last), 3);
                           std::transform(ldst, llast, ldst, [=](T const & v) { return op(prior_result, v); });
                           tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, last), 3);
                       }));

        phase2_input_handle = std::move(phase2_result_handle);

    }

    // irregular final chunk
    auto llast = std::next(ldst, std::distance(first, end));

    std::promise<T> phase2_result_promise;
    std::future<T> phase2_result_handle = phase2_result_promise.get_future();

    task_complete_handles.push_back(
        std::async(std::launch::async,
                   [=,
                    prior_completion_handle = std::move(completion_handles[thread_count - 1]),
                    phase2_input_handle = std::move(phase2_input_handle),
                    phase2_result_promise = std::move(phase2_result_promise)
                       ]() mutable {
                       prior_completion_handle.get();
                       // phase 1: parallel exclusive scans on each partition
                       tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, end), 1);
                       T local_result = sequential_exclusive_scan(first, end, ldst);
                       tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, end), 1);
                       // store the accumulated result for the next partition
                       T prior_result = phase2_input_handle.get();
                       phase2_result_promise.set_value(op(prior_result, local_result));
                       // phase 2: update sequential scan using results from partitions 0..i-1
                       tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, end), 3);
                       std::transform(ldst, llast, ldst, [=](T const & v) { return op(prior_result, v); });
                       tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, end), 3);
                   }));

    // return futures for
    // 1) the cumulative sum through this chunk
    // 2) the "done" status of each task

    return std::make_pair(std::move(phase2_result_handle),
                          std::move(task_complete_handles));

}

template<typename T, typename FwdIter1, typename FwdIter2, typename Op = std::plus<T>>
std::pair<FwdIter2, T> exclusive_scan_mt(FwdIter1 start, FwdIter1 end, FwdIter2 dst, T init = T(), Op op = Op())
{
    std::size_t sz = std::distance(start, end);

/*
    if (sz < 200000)
    {
        // in my tests this is roughly the right cutoff point
        return sequential_exclusive_scan(start, end, dst, init, op);
    }
*/

    std::size_t partition_size = sz / thread_count;
//    std::cout << "using partition size " << partition_size << "\n";

    // We do this in two phases contained in the same task, for cache locality reasons -
    // after the first phase the partition will be (at least in part) in the same core
    std::vector<std::promise<T>> phase2_input_handles(thread_count);

    // the "carry in" to the first group is already available - it's our init parameter
    phase2_input_handles[0].set_value(init);

    // We also need to clean up the tasks at the end
    std::vector<std::future<void>> task_complete_handles;
    task_complete_handles.reserve(thread_count - 1);

    FwdIter1 first = start;
    FwdIter2 ldst = dst;
    for (std::size_t i = 0;
         i < thread_count - 1;
         i++, std::advance(first, partition_size), std::advance(ldst, partition_size))
    {
        FwdIter1 last = std::next(first, partition_size);
        FwdIter2 llast = std::next(ldst, partition_size);

        task_complete_handles.push_back(
            std::async(std::launch::async,
                       [=, &phase2_input_handles](){
                           // phase 1: parallel exclusive scans on each partition
                           tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, last), 1);
                           T local_result = sequential_exclusive_scan(first, last, ldst);
                           tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, last), 1);
                           // store the accumulated result for the next partition
                           T prior_result = phase2_input_handles[i].get_future().get();
                           phase2_input_handles[i+1].set_value(op(prior_result, local_result));
                           // phase 2: update sequential scan using results from partitions 0..i-1
                           tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, last), 3);
                           std::transform(ldst, llast, ldst, [=](T const & v) { return op(prior_result, v); });
                           tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, last), 3);
                       }));
    }

    // the amount of data is not generally a multiple of the partition size so we
    // special case the final partition
    auto llast = std::next(ldst, std::distance(first, end));

    // run it directly
    tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, end), 1);
    T local_result = sequential_exclusive_scan(first, end, ldst);
    tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, end), 1);
    // store the accumulated result for the next "chunk"
    T prior_result = phase2_input_handles[thread_count - 1].get_future().get();
    T final_result = op(prior_result, local_result);
    tracepoint(HPX_ALG, chunk_start, std::distance(true_start, first), std::distance(true_start, end), 3);
    std::transform(ldst, llast, ldst, [=](T const & v) { return op(prior_result, v); });
    tracepoint(HPX_ALG, chunk_stop, std::distance(true_start, first), std::distance(true_start, end), 3);

    // phase 3: wait for completion of partition scans
    for (auto & f : task_complete_handles)
    {
        f.get();
    }


    return std::make_pair(llast, final_result);
}

namespace jet {
// chunk up the original data in cache-friendly sizes
template<typename T, typename FwdIter1, typename FwdIter2, typename Op = std::plus<T>>
FwdIter2 exclusive_scan(FwdIter1 start, FwdIter1 end, FwdIter2 dst, T init = T(), Op op = Op())
{
    std::size_t sz = std::distance(start, end);

    std::size_t chunk_count = (sz / chunksize) + 1;

    true_start = start;

    FwdIter1 last = std::next(start, std::min(chunksize, sz));

    // set up the futures for exchanging info between stages
    std::promise<T> init_promise;
    init_promise.set_value(init);
    std::future<T> running_sum_handle = init_promise.get_future();

    // completion futures - to keep work running predictably
    std::vector<std::promise<void>> chunk_completion_promises(thread_count);
    std::vector<std::future<void>> chunk_completion_handles;
    chunk_completion_handles.reserve(thread_count);
    for (std::promise<void> & p : chunk_completion_promises)
    {
        p.set_value();
        chunk_completion_handles.push_back(p.get_future());
    }


    for (std::size_t i = 0;
         i < (chunk_count - 1);
         ++i, std::advance(start, chunksize),
              std::advance(last, chunksize), std::advance(dst, chunksize))
    {
        // run the multi-threaded algorithm on the ith chunk
        std::tie(running_sum_handle, chunk_completion_handles) =
            exclusive_scan_mt_impl(start, last, dst, init, op,
                                   std::move(running_sum_handle),
                                   std::move(chunk_completion_handles));
    }

    // now the irregular final chunk
    std::tie(running_sum_handle, chunk_completion_handles) =
        exclusive_scan_mt_impl(start, end, dst, init, op,
                               std::move(running_sum_handle),
                               std::move(chunk_completion_handles));

    tracepoint(HPX_ALG, tasks_created);

    // wait for all tasks to complete
    for (std::future<void> & c : chunk_completion_handles)
    {
        c.get();
    }

    tracepoint(HPX_ALG, threads_done);

    return std::next(dst, std::distance(start, end));  // updating dst for final chunk

}
}

void verify()
{
    chunksize = 250000;   // just as long as it's not 0

    // generate a bunch of random cases to show it works
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> values_dist{0, 20};
    std::uniform_int_distribution<int> sizes_dist{1, 100000};

    const unsigned testcount = 10;

    for (unsigned t = 0; t < testcount; t++)
    {
//        auto sz = sizes_dist(mersenne_engine);
        auto sz = 40000000;
        std::vector<int> data(sz);
        std::generate(data.begin(), data.end(),
                      [&]() { return values_dist(mersenne_engine); });
        std::vector<int> par_result(sz);
        jet::exclusive_scan(data.begin(), data.end(), par_result.begin(), 1);
        std::vector<int> ser_result(sz);
        sequential_exclusive_scan(data.begin(), data.end(), ser_result.begin(), 1);
        assert(par_result == ser_result);

    }

}

// supply our custom thread_count/chunksize choices
static void CustomArguments(benchmark::internal::Benchmark* b) {
    for (int tc = 4; tc <=8; tc++)
        for (int chunk = 20000; chunk <= 200000; chunk += 10000)
            b->Args({16777216, tc, chunk});
}

int main(int argc, char* argv[])
{
    verify();
    std::exit(0);

    std::cout << "press enter to start benchmarking" << std::endl;
    getchar();

    // process and remove gbench arguments
    benchmark::Initialize(&argc, argv);

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> dist{1, 20};

    benchmark::RegisterBenchmark(
        "Sequential-STD",
        [&](benchmark::State & state)
        {
            auto sz = state.range(0);
            std::vector<int> data(sz);
            std::generate(data.begin(), data.end(),
                          [&]() { return dist(mersenne_engine); });
            std::vector<int> result(sz);
            for (auto _ : state) {
                state.PauseTiming();
                // flush input/output data to ensure we are "cold"
                for (int * pt = data.data(); pt < (data.data() + sz); pt += 16)
                    _mm_clflush(pt);
                for (int * pt = result.data(); pt < (result.data() + sz); pt += 16)
                    _mm_clflush(pt);
                state.ResumeTiming();

                sequential_exclusive_scan(data.begin(), data.end(), result.begin(), 0);
                benchmark::DoNotOptimize(result);
            }
        })->Range(10, 40000000)->UseRealTime();

    benchmark::RegisterBenchmark(
        "Parallel-STD",
        [&](benchmark::State & state)
        {
            auto sz = state.range(0);
            std::vector<int> data(sz);
            std::generate(data.begin(), data.end(),
                          [&]() { return dist(mersenne_engine); });
            std::vector<int> result(sz);
            thread_count = state.range(1);
            for (auto _ : state) {
                exclusive_scan_mt(data.begin(), data.end(), result.begin(), 0);
                benchmark::DoNotOptimize(result);
            }
        })->RangeMultiplier(2)->Ranges({{10, 40000000}, {1, 8}})->UseRealTime();   // size/threadcount

    benchmark::RegisterBenchmark(
        "Parallel-Chunked-STD",
        [&](benchmark::State & state)
        {
            auto sz = state.range(0);
            std::vector<int> data(sz);
            std::generate(data.begin(), data.end(),
                          [&]() { return dist(mersenne_engine); });
            std::vector<int> result(sz);
            thread_count = state.range(1);
            // we run the multi-thread algorithm on chunks of this size:
            chunksize = thread_count * state.range(2);
            // which means each thread gets a chunk of size state.range(2)
            // which is consistent with the way we analyze HPX
            for (auto _ : state) {
                state.PauseTiming();
                // flush input/output data to ensure we are "cold"
                for (int * pt = data.data(); pt < (data.data() + sz); pt += 16)
                    _mm_clflush(pt);
                for (int * pt = result.data(); pt < (result.data() + sz); pt += 16)
                    _mm_clflush(pt);
                state.ResumeTiming();

                tracepoint(HPX_ALG, benchmark_exe_start, 0);
                jet::exclusive_scan(data.begin(), data.end(), result.begin(), 0);
                tracepoint(HPX_ALG, benchmark_exe_stop);
                benchmark::DoNotOptimize(result);
            }
        })->Apply(CustomArguments)->UseRealTime();


    benchmark::RunSpecifiedBenchmarks();

}
