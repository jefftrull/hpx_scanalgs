// Trying to reproduce HPX exclusive_scan performance issues
// Author: Jeff Trull <edaskel@att.net>

#include <random>

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/parallel_algorithm.hpp>

#include <benchmark/benchmark.h>

#include <stdio.h>   // for getchar(). UGH

#include "tracepoints.h"

//
// special executor parameters for benchmarking
//

// special customization to override restriction on chunk count
struct unlimited_number_of_chunks
{
    template <typename Executor>
    std::size_t maximal_number_of_chunks(
        Executor&& executor, std::size_t cores, std::size_t num_tasks)
    {
        return num_tasks;
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_executor_parameters<unlimited_number_of_chunks> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

int main(int argc, char* argv[])
{
    // process and remove gbench arguments
    benchmark::Initialize(&argc, argv);

    // By default this should run on all available cores
    std::vector<std::string> const cfg = {};
/*
        "hpx.os_threads=cores"
    };
*/

    std::cout << "press enter to start benchmarking" << std::endl;
    getchar();

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
}

// setter for size/chunksize options
void
sz_range_setter(benchmark::internal::Benchmark* b)
{
    // exponential in overall size
    for (int sz = 262144; sz <= 40000000; sz *= 2)
        // linear in chunksize
        for (int csz = 20000; csz <= 200000; csz += 10000)
            if (csz < sz)
                b->Args({sz, csz});
}

template<typename ExePolicy>
void exs_bench(ExePolicy & ex, char const * name)
{
    using namespace hpx::parallel;
    using namespace hpx::util;

    benchmark::RegisterBenchmark(name, [&](benchmark::State &state) {
      // create and fill random vector of desired size
      std::random_device rnd_device;
      std::mt19937 mersenne_engine{rnd_device()};
      std::uniform_int_distribution<int> dist{1, 20};

      auto sz = state.range(0);
      std::vector<int> data(sz);
      std::generate(data.begin(), data.end(),
                    [&]() { return dist(mersenne_engine); });
      std::vector<int> result(sz + 1);

      // add chunksize parameter
      auto ex_cs = ex.with(execution::static_chunk_size(state.range(1)),
                           unlimited_number_of_chunks());

      for (auto _ : state) {
          tracepoint(HPX_ALG, benchmark_exe_start);
          exclusive_scan(ex_cs, data.begin(), data.end(), result.begin(), 0);
          tracepoint(HPX_ALG, benchmark_exe_stop);
          benchmark::DoNotOptimize(result);
      }
    })->Apply(sz_range_setter)->UseRealTime();
}

// sequential version has no "chunksize"
template <>
void exs_bench<hpx::parallel::execution::sequenced_policy>(
    hpx::parallel::execution::sequenced_policy const & ex,
    char const * name)
{
    using namespace hpx::parallel;
    using namespace hpx::util;

    benchmark::RegisterBenchmark(name, [&](benchmark::State &state) {
      // create and fill random vector of desired size
      std::random_device rnd_device;
      std::mt19937 mersenne_engine{rnd_device()};
      std::uniform_int_distribution<int> dist{1, 20};

      auto sz = state.range(0);
      std::vector<int> data(sz);
      std::generate(data.begin(), data.end(),
                    [&]() { return dist(mersenne_engine); });
      std::vector<int> result(sz + 1);

      // use executor unmodified
      for (auto _ : state) {
          state.PauseTiming();
          // flush input/output data to ensure we are "cold"
          for (int * pt = data.data(); pt < (data.data() + sz); pt += 16)
              _mm_clflush(pt);
          for (int * pt = result.data(); pt < (result.data() + sz); pt += 16)
              _mm_clflush(pt);
          state.ResumeTiming();

          exclusive_scan(ex, data.begin(), data.end(), result.begin(), 0);
          benchmark::DoNotOptimize(result);
          benchmark::ClobberMemory();
      }
    })->RangeMultiplier(2)->Range(262144, 33554432)->UseRealTime();
}

int hpx_main(int argc, char **argv)
{
    // process commandline options

    using namespace hpx::parallel;
    using namespace hpx::util;

    exs_bench(execution::seq, "Sequential");
    exs_bench(execution::par, "Parallel");

    benchmark::RunSpecifiedBenchmarks();

    return hpx::finalize();
}

