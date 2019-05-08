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

#include <benchmark/benchmark.h>

template<typename T, typename FwdIter1, typename FwdIter2, typename Op = std::plus<T>>
FwdIter2 sequential_exclusive_scan(FwdIter1 start, FwdIter1 end, FwdIter2 dst, T init, Op op = Op())
{
    while (start != end) {
        *dst++ = init;
        init = op(init, *start++);
    }
    return dst;
}

template<typename T, typename FwdIter1, typename FwdIter2, typename Op = std::plus<T>>
FwdIter2 exclusive_scan(FwdIter1 start, FwdIter1 end, FwdIter2 dst, T init = T(), Op op = Op())
{
    std::size_t sz = std::distance(start, end);

/*
    if (sz < 200000)
    {
        // in my tests this is roughly the right cutoff point
        return sequential_exclusive_scan(start, end, dst, init, op);
    }
*/

    // 2 threads isn't useful (same work, no overlap)
    const std::size_t thread_count = 4;   // this seems like the sweet spot with the current code

    std::size_t partition_size = sz / thread_count;
//    std::cout << "using partition size " << partition_size << "\n";

    // We do this in two phases contained in the same task, for cache locality reasons -
    // after the first phase the partition will be (at least in part) in the same core
    std::vector<std::promise<T>> phase2_input_handles(thread_count);

    // the "carry in" to the first group is already available - it's our init parameter
    phase2_input_handles[0].set_value(init);

    // we also need to clean up the tasks at the end
    std::vector<std::future<void>> task_complete_handles;

    FwdIter1 first = start;
    FwdIter2 ldst = dst;
    for (std::size_t i = 0;
         i < thread_count - 1;
         i++, std::advance(first, partition_size), std::advance(ldst, partition_size))
    {
        FwdIter1 last = first;
        std::advance(last, partition_size);

        task_complete_handles.push_back(
            std::async(std::launch::async,
                       [=, &phase2_input_handles](){
                           // phase 1: parallel accumulates on each partition
//                           std::cout << "launching accumulate " << i << "\n";
                           T local_result = std::accumulate(first, last, T{}, op);
                           // store the accumulated result for the next partition
                           T prior_result = phase2_input_handles[i].get_future().get();
                           phase2_input_handles[i+1].set_value(op(prior_result, local_result));
                           // phase 2: sequential scan using results from partitions 0..i-1
//                           std::cout << "launching exclusive_scan " << i << "\n";
                           sequential_exclusive_scan(first, last, ldst, prior_result, op);
                       }));
    }

    // the amount of data is not generally a multiple of the partition size so we
    // special case the final partition
    task_complete_handles.push_back(
        std::async(std::launch::async,
                   [=, &phase2_input_handles](){
/*
                           std::cout << "launching accumulate " << (thread_count - 1) << "\n";
*/
                       T prior_result = phase2_input_handles[thread_count - 1].get_future().get();
//                           std::cout << "launching exclusive_scan " << (thread_count - 1) << "\n";
                       sequential_exclusive_scan(first, end, ldst, prior_result, op);
                   }));

    // phase 3: wait for completion of partition scans
    for (auto & f : task_complete_handles)
    {
        f.get();
    }

    return end;
}

void verify()
{
    // generate a bunch of random cases to show it works
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> values_dist{0, 20};
    std::uniform_int_distribution<int> sizes_dist{1, 100000};

    const unsigned testcount = 10000;

    for (unsigned t = 0; t < testcount; t++)
    {
        auto sz = sizes_dist(mersenne_engine);
        std::vector<int> data(sz);
        std::generate(data.begin(), data.end(),
                      [&]() { return values_dist(mersenne_engine); });
        std::vector<int> par_result(sz);
        exclusive_scan(data.begin(), data.end(), par_result.begin(), 1);
        std::vector<int> ser_result(sz);
        sequential_exclusive_scan(data.begin(), data.end(), ser_result.begin(), 1);
        assert(par_result == ser_result);

    }

}

int main(int argc, char* argv[])
{
/*
    verify();
    std::exit(0);
*/

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
            for (auto _ : state) {
                exclusive_scan(data.begin(), data.end(), result.begin(), 0);
                benchmark::DoNotOptimize(result);
            }
        })->Range(10, 40000000)->UseRealTime();


    benchmark::RunSpecifiedBenchmarks();

}
