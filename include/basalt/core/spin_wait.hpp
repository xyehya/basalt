// include/basalt/core/spin_wait.hpp - one check of a host spin wait that gives its core away after a while.
//
// A thread that spins without ever yielding keeps its core: a thread woken onto that core (the job the spinner waits
// for, or the spinner's own waker) runs only at the scheduler's next tick.  A wait pauses for its first kYieldAfter,
// as the pool's workers do, then yields the core on every check.
#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

#include "basalt/platform/compiler.hpp"

namespace basalt::core {

class SpinWait {
public:
    static constexpr std::chrono::microseconds kYieldAfter{50};

    SpinWait() : t0_(std::chrono::steady_clock::now()) {}

    /// One check: a pause, or a yield once the wait has run past kYieldAfter.  The clock is read every 64 checks.
    void operator()() {
        if (yielding_) {
            std::this_thread::yield();
            return;
        }
        basalt::platform::cpu_relax();
        if ((++n_ & 63u) == 0 && std::chrono::steady_clock::now() - t0_ >= kYieldAfter) yielding_ = true;
    }

private:
    std::chrono::steady_clock::time_point t0_;
    bool yielding_ = false;
    uint32_t n_ = 0;
};

}  // namespace basalt::core
