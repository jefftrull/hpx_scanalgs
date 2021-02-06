// A probably-not-very-good thread pool, for evaluating manual parallelism
//  Copyright (c) 2021 Jeffrey Trull
//  and distributed under the MIT license

#include <future>
#include <thread>
#include <vector>
#include <queue>

struct thread_pool
{
    thread_pool(std::size_t sz);

    // What if f is move-only?
    // What if it takes arguments?
    template<typename F>
    std::future<std::invoke_result_t<F>>
    submit(F f);

    ~thread_pool();

    // become one of the worker threads until task queue empties
    void help();

private:
    void do_work();         // one call per thread

    std::size_t sz_;
    std::atomic<bool> shutting_down_;
    std::queue<std::packaged_task<void()>> tasks_;
    std::vector<std::thread> workers_;
    std::mutex check_for_work_mut_;
    std::condition_variable check_for_work_;
    
};

thread_pool::thread_pool(std::size_t sz)
    : sz_(sz), shutting_down_(false),
      workers_(sz_)
{
    for (std::size_t i = 0; i < sz_; ++i)
        workers_[i] = std::thread([this](){ do_work(); });
}

template<typename F>
std::future<std::invoke_result_t<F>>
thread_pool::submit(F f)
{
    if (shutting_down_.load())
        // draining queue
        // note this will get an exception if the consumer tries to use it
        // I suppose this is better than a deadlock?
        return std::future<std::invoke_result_t<F>>();

    using return_t = std::invoke_result_t<F>;
    std::promise<return_t> p;
    std::future<return_t> fut = p.get_future();

    {
        std::unique_lock<std::mutex> l(check_for_work_mut_);
        tasks_.emplace([p = std::move(p), f = std::move(f)]() mutable
        {
            // we should propagate exceptions here, oh well, maybe later
            if constexpr (!std::is_same_v<void, return_t>)
                p.set_value(f());
            else
            {
                f();
                p.set_value();
            }
        });
    }
    check_for_work_.notify_one();

    return fut;
}

thread_pool::~thread_pool()
{
    shutting_down_.store(true);

    check_for_work_.notify_all();

    for (auto & t : workers_)
        t.join();

}

void
thread_pool::do_work()
{
    // these repeated checks of the atomic can probably be cleaned up
    while (!shutting_down_.load())
    {
        std::packaged_task<void()> work;
        {
            std::unique_lock<std::mutex> l(check_for_work_mut_);
            check_for_work_.wait(l, [this]()
            {
                return shutting_down_.load() || !tasks_.empty();
            });

            if (!shutting_down_.load())
            {
                work = std::move(tasks_.front());
                tasks_.pop();
            }
        }

        if (!shutting_down_.load())
            work();
    }
}

void thread_pool::help()
{
    while (!shutting_down_.load())
    {
        std::packaged_task<void()> work;
        {
            std::unique_lock<std::mutex> l(check_for_work_mut_);

            if (tasks_.empty())
                return;

            work = std::move(tasks_.front());
            tasks_.pop();
        }

        work();
    }
}
