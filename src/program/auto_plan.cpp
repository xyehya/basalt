// src/program/auto_plan.cpp - see auto_plan.hpp.
#include "basalt/program/auto_plan.hpp"

#include <algorithm>
#include <cstdio>

namespace basalt::program {

namespace {
constexpr double kGiB = 1073741824.0;

bool fits(uint64_t complement, const AutoInputs& in) {
    return in.ram_available >= in.floor_bytes && complement <= in.ram_available - in.floor_bytes;
}
}  // namespace

const char* auto_affinity_name(AutoAffinity a) {
    return a == AutoAffinity::All ? "all" : a == AutoAffinity::Auto ? "auto" : "p-cores";
}

int auto_pool_workers(const AutoInputs& in, AutoAffinity affinity) {
    int n = in.worker_cores;
    if (in.hybrid && affinity != AutoAffinity::All) {
        // The P-cores but the host's. The auto affinity adds half the E-cores: they run the expert kernels slower,
        // but a layer's CPU share is split into many small tasks, so each one still shortens it.
        n = (std::max)(1, in.p_cores - 1);
        if (affinity == AutoAffinity::Auto) n += in.e_cores / 2;
    }
    // A CPU quota throttles the whole group once spent: the workers and the host thread together must fit it.
    if (in.cpu_quota > 0) {
        const int fit = (std::max)(1, (int) in.cpu_quota - 1);
        if (fit < n) n = fit;
    }
    return (std::max)(1, n);
}

AutoPlan choose_auto(const AutoInputs& in) {
    const AutoGiven& g = in.given;
    AutoPlan p;

    // Peer-exclusive only when the full complement does not fit and holding the peer's experts only on the peer
    // leaves nothing to read from the SSD.
    if (g.peer_exclusive) {
        p.peer_exclusive = g.peer_exclusive_value;
        p.peer_exclusive_src = AutoSource::Given;
    } else {
        p.peer_exclusive = in.peer && in.serve && !fits(in.complement_full, in) && fits(in.complement_peer_excl, in);
    }

    // Resident: what fits, capped at the complement.
    if (g.resident) {
        p.resident_gb = g.resident_gb;
        p.resident_bytes = (uint64_t) (g.resident_gb * kGiB);
        p.resident_src = AutoSource::Given;
    } else {
        const uint64_t complement = p.peer_exclusive ? in.complement_peer_excl : in.complement_full;
        const uint64_t room = in.ram_available > in.floor_bytes ? in.ram_available - in.floor_bytes : 0;
        p.resident_bytes = (std::min)(room, complement);
        p.resident_gb = (double) p.resident_bytes / kGiB;
    }

    if (g.pool_affinity) {
        p.pool_affinity = g.pool_affinity_value;
        p.pool_affinity_src = AutoSource::Given;
    } else {
        p.pool_affinity = AutoAffinity::Auto;
    }

    if (g.pool_workers && g.pool_workers_value > 0) {
        p.pool_workers = g.pool_workers_value;
        p.pool_workers_src = AutoSource::Given;
    } else {
        p.pool_workers = auto_pool_workers(in, p.pool_affinity);
    }
    return p;
}

KvHostChoice choose_kv_host(const KvHostInputs& in) {
    KvHostChoice c;
    if (in.cells <= 0 || in.bytes_per_cell == 0 || in.cells < in.crossover_cells) {
        c.why = "shallow";
        return c;
    }
    const uint64_t pin = (uint64_t) in.cells * in.bytes_per_cell;
    if (in.ram_available <= in.ram_kept || pin > in.ram_available - in.ram_kept) {
        c.why = "no RAM";
        return c;
    }
    c.host = true;
    c.pin_bytes = pin;
    c.why = "host";
    return c;
}

int64_t kv_host_cells(int64_t need, int64_t max_new, int64_t slack, int64_t grid, int64_t max_cells) {
    int64_t c = std::max<int64_t>(need, 0) + std::max<int64_t>(max_new, 0) + std::max<int64_t>(slack, 0);
    if (grid > 0) c = (c + grid - 1) / grid * grid;
    return max_cells > 0 ? std::min(c, max_cells) : c;
}

KvHostPlan plan_kv_host(const KvHostInputs& in, int64_t pool_cells) {
    KvHostPlan p;
    const bool deep = in.cells > 0 && in.bytes_per_cell > 0 && in.cells >= in.crossover_cells;
    if (pool_cells > 0 && deep && pool_cells >= in.cells) {
        p.host = true;
        p.act = KvHostAct::keep;
        p.cells = pool_cells;
        p.why = "host";
        return p;
    }
    const KvHostChoice c = choose_kv_host(in);
    p.why = c.why;
    if (c.host) {
        p.host = true;
        p.act = pool_cells > 0 ? KvHostAct::grow : KvHostAct::take;
        p.cells = in.cells;
        p.pin_bytes = c.pin_bytes;
        return p;
    }
    p.act = pool_cells > 0 ? KvHostAct::drop : KvHostAct::none;
    return p;
}

const char* kv_host_act_name(KvHostAct a) {
    switch (a) {
        case KvHostAct::take: return "take";
        case KvHostAct::keep: return "keep";
        case KvHostAct::grow: return "grow";
        case KvHostAct::drop: return "drop";
        default: return "none";
    }
}

std::string auto_flags_line(const AutoPlan& p) {
    std::string out;
    auto add = [&](const std::string& s) { out += out.empty() ? "auto flags: " : ", "; out += s; };
    char buf[64];
    if (p.peer_exclusive_src == AutoSource::Auto) add(std::string("peer-exclusive ") + (p.peer_exclusive ? "on" : "off"));
    if (p.resident_src == AutoSource::Auto) {
        std::snprintf(buf, sizeof buf, "resident %.1f GB", p.resident_gb);
        add(buf);
    }
    if (p.pool_affinity_src == AutoSource::Auto) add(std::string("pool-affinity ") + auto_affinity_name(p.pool_affinity));
    if (p.pool_workers_src == AutoSource::Auto) {
        std::snprintf(buf, sizeof buf, "pool-workers %d", p.pool_workers);
        add(buf);
    }
    return out;
}

}  // namespace basalt::program
