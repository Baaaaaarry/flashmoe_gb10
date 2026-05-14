#include "flashmoe/io_pool.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <unistd.h>

namespace flashmoe {

ChunkedPreadPool::ChunkedPreadPool(std::size_t worker_count) {
    workers_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ChunkedPreadPool::~ChunkedPreadPool() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ChunkedPreadPool::submit(const IoTask& task) {
    {
        std::lock_guard lock(mutex_);
        queue_.push(task);
        in_flight_ += 1;
    }
    cv_.notify_one();
}

void ChunkedPreadPool::submit_chunked(int fd,
                                      off_t offset,
                                      std::size_t size,
                                      void* dst,
                                      std::size_t split,
                                      std::size_t page_size,
                                      std::function<void(ssize_t)> on_complete) {
    if (split <= 1 || page_size == 0 || (size % page_size) != 0) {
        submit(IoTask{fd, offset, size, dst, std::move(on_complete)});
        return;
    }

    split = std::min<std::size_t>(split, size / page_size);
    const std::size_t total_pages = size / page_size;
    auto remaining = std::make_shared<std::atomic<std::size_t>>(split);
    auto aggregate = std::make_shared<std::atomic<ssize_t>>(0);

    std::size_t page_cursor = 0;
    for (std::size_t chunk = 0; chunk < split; ++chunk) {
        std::size_t pages_this_chunk = total_pages / split;
        if (chunk < (total_pages % split)) {
            pages_this_chunk += 1;
        }
        const std::size_t chunk_offset = page_cursor * page_size;
        const std::size_t chunk_size = pages_this_chunk * page_size;
        page_cursor += pages_this_chunk;

        submit(IoTask{
            .fd = fd,
            .offset = static_cast<off_t>(offset + static_cast<off_t>(chunk_offset)),
            .size = chunk_size,
            .dst = static_cast<char*>(dst) + chunk_offset,
            .on_complete = [remaining, aggregate, on_complete](ssize_t result) {
                if (result > 0) {
                    aggregate->fetch_add(result);
                } else {
                    aggregate->store(result);
                }
                if (remaining->fetch_sub(1) == 1 && on_complete) {
                    on_complete(aggregate->load());
                }
            },
        });
    }
}

void ChunkedPreadPool::drain() {
    std::unique_lock lock(mutex_);
    drained_cv_.wait(lock, [this] { return in_flight_ == 0 && queue_.empty(); });
}

void ChunkedPreadPool::worker_loop() {
    while (true) {
        IoTask task;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_ && queue_.empty()) {
                return;
            }
            task = queue_.front();
            queue_.pop();
        }

        const ssize_t result = ::pread(task.fd, task.dst, task.size, task.offset);
        if (task.on_complete) {
            task.on_complete(result);
        }

        {
            std::lock_guard lock(mutex_);
            if (in_flight_ > 0) {
                in_flight_ -= 1;
            }
            if (in_flight_ == 0 && queue_.empty()) {
                drained_cv_.notify_all();
            }
        }
    }
}

}  // namespace flashmoe
