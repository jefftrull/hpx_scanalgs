# hpx_scanalgs
Experimenting with parallel scan (inclusive/exclusive) algorithms with a focus on improving those in HPX

## To Build and Run Benchmarks

```
mkdir build;cd build
cmake -DBOOST_ROOT=/path/to/boost -DHPX_DIR=/root/of/hpx/lib/cmake/HPX ..
make
./exsvp # for HPX's exclusive_scan implementation
./mp    # for my std library-based implementation
