// Trying to reproduce HPX exclusive_scan performance issues
// Author: Jeff Trull <edaskel@att.net>

#include <random>

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/parallel_algorithm.hpp>

#include <benchmark/benchmark.h>

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

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
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
      for (auto _ : state) {
        exclusive_scan(ex, data.begin(), data.end(), result.begin(), 0);
        benchmark::DoNotOptimize(result);
      }
    })->Range(10, 40000000);
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

