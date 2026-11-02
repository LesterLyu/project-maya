// Windows portable host registration policy, adapted from Basalt's pin_budget.
// The RAM allocation and its CUDA registration have independent budgets.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#endif

namespace strata::platform {

inline uint64_t host_pin_allowance(uint64_t budget, uint64_t used, uint64_t available,
                                   uint64_t reservation, bool shared) {
    if (shared) {
        available = std::min(available, budget);
        budget = available + std::min(reservation, budget - available);
    }
    const uint64_t reserve = std::max<uint64_t>(budget / 20, uint64_t(1) << 30);
    const uint64_t free = budget > used ? budget - used : 0;
    return free > reserve ? free - reserve : 0;
}

inline bool host_pin_range(uintptr_t base, uint64_t size, uintptr_t pointer, uint64_t bytes) {
    return pointer >= base && uint64_t(pointer - base) <= size && bytes <= size - uint64_t(pointer - base);
}

// Additional registration allowed on this adapter. A zero allowance is a known
// exhausted budget, distinct from an unavailable query. No static dxgi link.
inline bool nonlocal_pin_budget(const void* luid, uint32_t node_mask, bool shared,
                               uint64_t& allowance, std::string& why) {
#if defined(_WIN32)
    if (luid == nullptr || node_mask == 0) return false;
    HMODULE module = LoadLibraryA("dxgi.dll");
    if (module == nullptr) return false;
    using Create = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = reinterpret_cast<Create>(reinterpret_cast<void*>(GetProcAddress(module, "CreateDXGIFactory1")));
    IDXGIFactory1* factory = nullptr;
    bool found = false;
    if (create != nullptr && SUCCEEDED(create(__uuidof(IDXGIFactory1), (void**) &factory))) {
        IDXGIAdapter1* adapter = nullptr;
        for (UINT i = 0; !found && factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            IDXGIAdapter3* query = nullptr;
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && std::memcmp(&desc.AdapterLuid, luid, 8) == 0 &&
                SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &query))) {
                for (UINT node = 0; node < 32; ++node) {
                    if (!(node_mask & (uint32_t(1) << node))) continue;
                    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
                    if (FAILED(query->QueryVideoMemoryInfo(node, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) continue;
                    const uint64_t remaining = host_pin_allowance(info.Budget, info.CurrentUsage,
                        info.AvailableForReservation, info.CurrentReservation, shared);
                    if (!found || remaining < allowance) {
                        allowance = remaining;
                        char message[192];
                        std::snprintf(message, sizeof message,
                            "DXGI non-local budget %.2f GiB, used %.2f GiB, available registration %.2f GiB",
                            double(info.Budget) / 1073741824.0, double(info.CurrentUsage) / 1073741824.0,
                            double(remaining) / 1073741824.0);
                        why = message;
                    }
                    found = true;
                }
                query->Release();
            }
            adapter->Release();
        }
        factory->Release();
    }
    FreeLibrary(module);
    return found;
#else
    (void) luid; (void) node_mask; (void) shared; (void) allowance; (void) why;
    return false;
#endif
}
}  // namespace strata::platform
