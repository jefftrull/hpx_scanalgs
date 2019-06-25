// Benchmarking core algorithms for exclusive_scan

#include <algorithm>
#include <numeric>
#include <random>

#include <benchmark/benchmark.h>

int main(int argc, char *argv[]) {
    // process and remove gbench arguments
    benchmark::Initialize(&argc, argv);

    std::random_device rnd_device;
    std::mt19937 mersenne_engine{rnd_device()};
    std::uniform_int_distribution<int> dist{1, 20};

    benchmark::RegisterBenchmark(
        "std::accumulate",
        [&](benchmark::State & state)
        {
            auto sz = state.range(0);
            std::vector<int> data(sz);
            std::generate(data.begin(), data.end(),
                          [&]() { return dist(mersenne_engine); });
            int result;
            for (auto _ : state)
            {
                result = std::accumulate(data.begin(), data.end(), 0);
                benchmark::DoNotOptimize(result);
            }
        })->RangeMultiplier(10)->Range(10, 100000000);

    benchmark::RegisterBenchmark(
        "std::partial_sum",
        [&](benchmark::State & state)
        {
            auto sz = state.range(0);
            std::vector<int> data(sz);
            std::generate(data.begin(), data.end(),
                          [&]() { return dist(mersenne_engine); });
            std::vector<int> result(sz);
            for (auto _ : state)
            {
                std::partial_sum(data.begin(), data.end(), result.begin());
                benchmark::DoNotOptimize(result);
            }
        })->RangeMultiplier(10)->Range(10, 100000000);

    // testing the benefits of a hot cache
    benchmark::RegisterBenchmark(
        "acc then ps",
        [&](benchmark::State & state)
        {
            auto sz = state.range(0);
            std::vector<int> data(sz);
            std::generate(data.begin(), data.end(),
                          [&]() { return dist(mersenne_engine); });
            int sum;
            std::vector<int> result(sz);
            for (auto _ : state)
            {
                sum = std::accumulate(data.begin(), data.end(), 0);
                std::partial_sum(data.begin(), data.end(), result.begin());
                benchmark::DoNotOptimize(sum);
                benchmark::DoNotOptimize(result);
            }
        })->RangeMultiplier(10)->Range(10, 100000000);
    // appears to be worse!?


    benchmark::RunSpecifiedBenchmarks();
}
