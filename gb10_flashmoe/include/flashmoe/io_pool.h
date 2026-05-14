#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace flashmoe {

struct IoTask {
    int fd = -1;
    off_t offset = 0;
    std::size_t size = 0;
    void* dst = nullptr;
    std::function<void(ssize_t)> on_complete;
};

class ChunkedPreadPool {
public:
    explicit ChunkedPreadPool(std::size_t worker_count);
    ~ChunkedPreadPool();

    ChunkedPreadPool(const ChunkedPreadPool&) = delete;
    ChunkedPreadPool& operator=(const ChunkedPreadPool&) = delete;

    void submit(const IoTask& task);
    void submit_chunked(int fd,
                        off_t offset,
                        std::size_t size,
                        void* dst,
                        std::size_t split,
                        std::size_t page_size,
                        std::function<void(ssize_t)> on_complete);
    void drain();

private:
    void worker_loop();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable drained_cv_;
    std::queue<IoTask> queue_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
    std::size_t in_flight_ = 0;
};

}  // namespace flashmoe
