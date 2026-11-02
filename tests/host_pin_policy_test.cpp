#include "strata/platform/host_pin.hpp"
#include <cstdio>

int main() {
    using strata::platform::host_pin_allowance;
    using strata::platform::host_pin_range;
    constexpr uint64_t GiB = uint64_t(1) << 30;
    auto check = [](bool ok, const char* message) { if (!ok) std::fprintf(stderr, "%s\n", message); return ok; };
    bool ok = true;
    ok &= check(host_pin_allowance(64*GiB, 4*GiB, 64*GiB, 0, false) == 64*GiB-4*GiB-64*GiB/20,
                "single adapter keeps usage and five percent reserve");
    ok &= check(host_pin_allowance(64*GiB, 4*GiB, 32*GiB, 0, true) == 32*GiB-4*GiB-32*GiB/20,
                "portable mappings obey reservable capacity");
    ok &= check(host_pin_allowance(64*GiB, 64*GiB, 32*GiB, 0, true) == 0,
                "exhausted usage cannot underflow");
    ok &= check(host_pin_allowance(GiB/2, 0, GiB/2, 0, false) == 0,
                "small adapters retain the one GiB reserve floor");
    ok &= check(host_pin_allowance(4*GiB, 0, 8*GiB, 8*GiB, true) == 3*GiB,
                "reservation accounting cannot exceed the real budget");
    ok &= check(host_pin_range(100, 100, 150, 50), "registered range end is covered");
    ok &= check(!host_pin_range(100, 100, 150, 51), "unregistered tail cannot DMA directly");
    ok &= check(!host_pin_range(100, 100, 99, 1), "preceding bytes cannot DMA directly");
    if (ok) std::puts("PASS: host pin allowance and registered-range boundaries");
    return ok ? 0 : 1;
}
