#pragma once

#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <unordered_map>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h> // For SIMD operations (x86/x64 only)
#endif

// Invariants:
// - Dtype: currently supports float32 only (specialized path)
// - Shapes: up_weight [sparse_dim, up_cols]; down_weight [hidden_size, sparse_dim];
//           gate_weight (optional) [sparse_dim, gate_cols]
// - Mask: boolean, length sparse_dim; maintained on current_device
// - Outputs: get_concat_weight() -> [num_active, (gate_cols + up_cols) or up_cols]
//            get_active_down_weight() -> [hidden_dim, num_active]
// - Zero-copy: on CPU, returned tensors alias internal buffers; on non-CPU, tensors are materialized on device
class WeightCache : public torch::CustomClassHolder
{
private:
    // Define deleter as a struct to avoid std::function overhead
    struct AlignedDeleter
    {
        void operator()(float *ptr) const
        {
            free(ptr);
        }
    };

    bool is_initialized = false;
    bool has_gate_buffers;

    // Memory pools for all weight data (cache-aligned)
    std::unique_ptr<float[], AlignedDeleter> gate_memory_pool;
    std::unique_ptr<float[], AlignedDeleter> up_memory_pool;
    std::unique_ptr<float[], AlignedDeleter> down_memory_pool_transposed; // Store transposed for fast row access

    // Matrix dimensions
    int64_t hidden_dim = 0;
    int64_t sparse_dim = 0;
    int64_t gate_row_size = 0;
    int64_t up_row_size = 0;
    int64_t down_row_size = 0; // This becomes hidden_dim after transpose

    torch::ScalarType dtype;
    torch::Device current_device = torch::kCPU;

    // Currently active indices (maintained in order for contiguous access)
    std::vector<int64_t> active_indices;

    // Mapping from active_index to position in active_indices for O(1) lookup
    std::unordered_map<int64_t, size_t> index_to_position;

    // Contiguous buffers for active data (always packed)
    std::unique_ptr<float[], AlignedDeleter> active_gate_buffer;
    std::unique_ptr<float[], AlignedDeleter> active_up_buffer;
    std::unique_ptr<float[], AlignedDeleter> active_down_buffer;

    // Current mask for differential updates
    torch::Tensor current_mask;

    // Cached active weight tensors - use from_blob to reference our buffers directly
    torch::Tensor active_weights_cache;
    torch::Tensor active_downs_cache;
    bool cache_valid = false;

    // Max expected active indices (dynamic based on intermediate_size)
    size_t max_active_indices = 0;
    size_t min_active_indices = 0;

    // Environment-configurable parallelization knobs
    static int64_t env_as_int(const char *name, int64_t def) {
        const char *s = std::getenv(name);
        if (!s) return def;
        char *end = nullptr;
        long long v = std::strtoll(s, &end, 10);
        if (end == s) return def;
        return static_cast<int64_t>(v);
    }

    // Cache-aligned memory allocation
    static void *aligned_alloc_wrapper(size_t size)
    {
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 64, size) != 0)
        { // 64-byte alignment for cache lines
            throw std::bad_alloc();
        }
        return ptr;
    }

    // Find differential changes between masks using PyTorch operations
    struct MaskDiff
    {
        std::vector<int64_t> added_indices;
        std::vector<int64_t> removed_indices;
    };

    MaskDiff compute_mask_diff(const torch::Tensor &old_mask, const torch::Tensor &new_mask)
    {
        MaskDiff diff;

        // Use PyTorch operations for efficient mask comparison
        auto added_mask = new_mask & (~old_mask);   // new & ~old = added
        auto removed_mask = old_mask & (~new_mask); // old & ~new = removed

        // Get indices of added and removed elements
        // Ensure indices are on CPU before accessing data_ptr from host
        auto added_indices_tensor = torch::nonzero(added_mask)
                                         .squeeze(-1)
                                         .to(torch::kCPU)
                                         .contiguous();
        auto removed_indices_tensor = torch::nonzero(removed_mask)
                                           .squeeze(-1)
                                           .to(torch::kCPU)
                                           .contiguous();

        // Convert to std::vector
        if (added_indices_tensor.numel() > 0)
        {
            auto added_data = added_indices_tensor.data_ptr<int64_t>();
            diff.added_indices.assign(added_data, added_data + added_indices_tensor.numel());
        }

        if (removed_indices_tensor.numel() > 0)
        {
            auto removed_data = removed_indices_tensor.data_ptr<int64_t>();
            diff.removed_indices.assign(removed_data, removed_data + removed_indices_tensor.numel());
        }

        return diff;
    }

    // Rebuild tensors using from_blob to reference our contiguous buffers
    void rebuild_tensor_views()
    {
        const size_t num_active = active_indices.size();
        auto opts = torch::TensorOptions().dtype(dtype);

        // Support empty active set: return well-shaped empty tensors
        if (num_active == 0)
        {
            // Build empty logical views
            auto up_empty = torch::empty({0, up_row_size}, opts);
            torch::Tensor weights_empty;
            if (has_gate_buffers)
            {
                auto gate_empty = torch::empty({0, gate_row_size}, opts);
                weights_empty = torch::cat({gate_empty, up_empty}, 1);
            }
            else
            {
                weights_empty = up_empty;
            }

            auto down_empty = torch::empty({hidden_dim, 0}, opts);

            if (current_device.is_cpu())
            {
                active_weights_cache = weights_empty;
                active_downs_cache = down_empty;
            }
            else
            {
                active_weights_cache = weights_empty.to(current_device);
                active_downs_cache = down_empty.to(current_device);
            }
            return;
        }

        // Create up tensor directly from buffer
        auto up_tensor = torch::from_blob(active_up_buffer.get(),
                                          {static_cast<int64_t>(num_active), up_row_size},
                                          torch::TensorOptions().dtype(dtype));

        // Create down tensor directly from buffer and transpose
        auto down_tensor_packed = torch::from_blob(active_down_buffer.get(),
                                                   {static_cast<int64_t>(num_active), hidden_dim},
                                                   torch::TensorOptions().dtype(dtype));
        auto down_tensor = down_tensor_packed.t(); // [hidden_dim, num_active]

        // Create gate tensor directly from buffer
        if (has_gate_buffers)
        {
            auto gate_tensor = torch::from_blob(active_gate_buffer.get(),
                                                {static_cast<int64_t>(num_active), gate_row_size},
                                                torch::TensorOptions().dtype(dtype));

            if (current_device.is_cpu())
            {
                // Zero-copy on CPU: keep views over our buffers
                active_weights_cache = torch::cat({gate_tensor, up_tensor}, 1);
            }
            else
            {
                // Non-CPU target: materialize on device
                active_weights_cache = torch::cat({gate_tensor, up_tensor}, 1).to(current_device);
            }
        }
        else
        {
            if (current_device.is_cpu())
            {
                // Zero-copy on CPU
                active_weights_cache = up_tensor;
            }
            else
            {
                active_weights_cache = up_tensor.to(current_device);
            }
        }

        if (current_device.is_cpu())
        {
            // Zero-copy on CPU
            active_downs_cache = down_tensor;
        }
        else
        {
            active_downs_cache = down_tensor.to(current_device);
        }
    }

public:
    WeightCache(const torch::Tensor &init_mask, int64_t hidden_size,
                const torch::Tensor &gate_weight, const torch::Tensor &up_weight,
                const torch::Tensor &down_weight, bool has_gate=true)
    {
        init(init_mask, hidden_size, gate_weight, up_weight, down_weight, has_gate);
    }

    void init(const torch::Tensor &init_mask, int64_t hidden_size,
              const torch::Tensor &gate_weight, const torch::Tensor &up_weight,
              const torch::Tensor &down_weight, bool has_gate)
    {
        // Basic input validation for research readiness
        TORCH_CHECK(up_weight.dim() == 2, "up_weight must be 2D [sparse_dim, up_cols]");
        TORCH_CHECK(down_weight.dim() == 2, "down_weight must be 2D [hidden_size, sparse_dim]");
        if (has_gate)
        {
            TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be 2D [sparse_dim, gate_cols]");
        }

        // Enforce dtype compatibility with the current implementation (float32)
        TORCH_CHECK(up_weight.scalar_type() == torch::kFloat32,
                    "Only float32 supported currently for up_weight");
        TORCH_CHECK(down_weight.scalar_type() == torch::kFloat32,
                    "Only float32 supported currently for down_weight");
        if (has_gate)
        {
            TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32,
                        "Only float32 supported currently for gate_weight");
        }

        current_device = up_weight.device();
        dtype = up_weight.scalar_type();
        has_gate_buffers = has_gate;

        // Store dimensions
        hidden_dim = hidden_size;
        sparse_dim = up_weight.size(0);
        max_active_indices = sparse_dim;
        min_active_indices = 0;
        up_row_size = up_weight.size(1);
        down_row_size = hidden_dim; // After transpose: [intermediate_size, hidden_size]
        if (has_gate_buffers)
            gate_row_size = gate_weight.size(1);

        // Validate shape consistency
        TORCH_CHECK(down_weight.size(0) == hidden_dim && down_weight.size(1) == sparse_dim,
                    "down_weight must have shape [hidden_size, sparse_dim]");
        if (has_gate_buffers)
        {
            TORCH_CHECK(gate_weight.size(0) == sparse_dim,
                        "gate_weight first dim must equal up_weight first dim (sparse_dim)");
        }

        // Allocate cache-aligned memory pools
        const size_t gate_total_size = sparse_dim * gate_row_size;
        const size_t up_total_size = sparse_dim * up_row_size;
        const size_t down_total_size = sparse_dim * hidden_dim; // Transposed shape

        if (has_gate_buffers)
            gate_memory_pool = std::unique_ptr<float[], AlignedDeleter>(
                static_cast<float *>(aligned_alloc_wrapper(gate_total_size * sizeof(float))));
        up_memory_pool = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(up_total_size * sizeof(float))));
        down_memory_pool_transposed = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(down_total_size * sizeof(float))));

        // Pre-allocate contiguous buffers for active weights
        if (has_gate_buffers)
            active_gate_buffer = std::unique_ptr<float[], AlignedDeleter>(
                static_cast<float *>(aligned_alloc_wrapper(max_active_indices * gate_row_size * sizeof(float))));
        active_up_buffer = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(max_active_indices * up_row_size * sizeof(float))));
        active_down_buffer = std::unique_ptr<float[], AlignedDeleter>(
            static_cast<float *>(aligned_alloc_wrapper(max_active_indices * hidden_dim * sizeof(float))));

        // Initialize differential update tracking
        index_to_position.reserve(max_active_indices);

        // Copy weights to memory pools
        auto up_cpu = up_weight.to(torch::kCPU).contiguous();
        auto down_cpu = down_weight.to(torch::kCPU).contiguous();

        // Copy gate and up weights directly (row-major format)
        std::memcpy(up_memory_pool.get(), up_cpu.data_ptr<float>(), up_total_size * sizeof(float));
        if (has_gate_buffers)
        {
            auto gate_cpu = gate_weight.to(torch::kCPU).contiguous();
            std::memcpy(gate_memory_pool.get(), gate_cpu.data_ptr<float>(), gate_total_size * sizeof(float));
        }
        

        // Transpose down matrix during copy: [hidden_size, intermediate_size] -> [intermediate_size, hidden_size]
        auto down_data = down_cpu.data_ptr<float>();
        for (int64_t i = 0; i < sparse_dim; ++i)
        {
            for (int64_t j = 0; j < hidden_dim; ++j)
            {
                down_memory_pool_transposed[i * hidden_dim + j] = down_data[j * sparse_dim + i];
            }
        }

        is_initialized = true;
        current_mask = torch::zeros(sparse_dim, torch::TensorOptions().dtype(torch::kBool).device(current_device));

        // Initialize with mask
        update_active_weights(init_mask);
    }

    void update_active_weights(const torch::Tensor &mask)
    {
        if (!is_initialized)
            return;

        // Normalize mask to expected device and dtype (bool)
        TORCH_CHECK(mask.numel() == sparse_dim,
                    "mask length (numel) must equal sparse_dim");
        auto normalized_mask = mask.to(current_device, torch::kBool).view({-1}).contiguous();

        // Compute diff with normalized mask
        auto diff = compute_mask_diff(current_mask, normalized_mask);
        // Early exit if no changes - avoid all processing work!
        if (diff.added_indices.empty() && diff.removed_indices.empty())
        {
            return;
        }

        // PAIRED REPLACEMENT ALGORITHM - Core Innovation
        const size_t num_removals = diff.removed_indices.size();
        const size_t num_additions = diff.added_indices.size();
        const size_t pairs_to_process = std::min(num_removals, num_additions);

        // Phase 1: Pair removals with additions for direct replacement (most cache-efficient)
        // To enable parallel memcpy, precompute valid (pos, added_idx, removed_idx) tuples.
        std::vector<size_t> pair_positions;
        std::vector<int64_t> pair_added;
        std::vector<int64_t> pair_removed;
        pair_positions.reserve(pairs_to_process);
        pair_added.reserve(pairs_to_process);
        pair_removed.reserve(pairs_to_process);
        for (size_t i = 0; i < pairs_to_process; ++i)
        {
            int64_t removed_idx = diff.removed_indices[i];
            auto it = index_to_position.find(removed_idx);
            if (it != index_to_position.end())
            {
                pair_positions.push_back(it->second);
                pair_added.push_back(diff.added_indices[i]);
                pair_removed.push_back(removed_idx);
            }
        }

        // Parallel memcpy over pairs (metadata updates applied after)
        const int64_t num_pairs = static_cast<int64_t>(pair_positions.size());
        if (num_pairs > 0)
        {
            const int64_t grain = env_as_int("WEIGHT_CACHE_GRAIN", 64);
            const int64_t threshold = env_as_int("WEIGHT_CACHE_PAR_THRESHOLD", 64);
            if (num_pairs >= threshold)
            {
                at::parallel_for(0, num_pairs, grain, [&](int64_t begin, int64_t end)
                                 {
                                     for (int64_t i = begin; i < end; ++i)
                                     {
                                         size_t pos = pair_positions[i];
                                         int64_t added_idx = pair_added[i];
                                         if (has_gate_buffers)
                                             std::memcpy(active_gate_buffer.get() + pos * gate_row_size,
                                                         gate_memory_pool.get() + added_idx * gate_row_size,
                                                         gate_row_size * sizeof(float));

                                         std::memcpy(active_up_buffer.get() + pos * up_row_size,
                                                     up_memory_pool.get() + added_idx * up_row_size,
                                                     up_row_size * sizeof(float));

                                         std::memcpy(active_down_buffer.get() + pos * hidden_dim,
                                                     down_memory_pool_transposed.get() + added_idx * hidden_dim,
                                                     hidden_dim * sizeof(float));
                                     }
                                 });
            }
            else
            {
                for (int64_t i = 0; i < num_pairs; ++i)
                {
                    size_t pos = pair_positions[i];
                    int64_t added_idx = pair_added[i];
                    if (has_gate_buffers)
                        std::memcpy(active_gate_buffer.get() + pos * gate_row_size,
                                    gate_memory_pool.get() + added_idx * gate_row_size,
                                    gate_row_size * sizeof(float));

                    std::memcpy(active_up_buffer.get() + pos * up_row_size,
                                up_memory_pool.get() + added_idx * up_row_size,
                                up_row_size * sizeof(float));

                    std::memcpy(active_down_buffer.get() + pos * hidden_dim,
                                down_memory_pool_transposed.get() + added_idx * hidden_dim,
                                hidden_dim * sizeof(float));
                }
            }

            // Sequential metadata updates
            for (int64_t i = 0; i < num_pairs; ++i)
            {
                size_t pos = pair_positions[i];
                int64_t added_idx = pair_added[i];
                int64_t removed_idx = pair_removed[i];
                index_to_position.erase(removed_idx);
                active_indices[pos] = added_idx;
                index_to_position[added_idx] = pos;
            }
        }

        // Handle remaining additions (if more additions than removals)
        // Remaining additions: compute allowed count to avoid overflow
        if (num_additions > pairs_to_process)
        {
            const size_t rem_additions = num_additions - pairs_to_process;
            size_t start_pos = active_indices.size();
            size_t capacity_left = (max_active_indices > start_pos) ? (max_active_indices - start_pos) : 0;
            size_t to_append = std::min(rem_additions, capacity_left);
            // Parallel memcpy for additions
            const int64_t grain2 = env_as_int("WEIGHT_CACHE_GRAIN", 64);
            const int64_t threshold2 = env_as_int("WEIGHT_CACHE_PAR_THRESHOLD", 64);
            if (static_cast<int64_t>(to_append) >= threshold2)
            {
                at::parallel_for(0, static_cast<int64_t>(to_append), grain2, [&](int64_t begin, int64_t end)
                                 {
                                     for (int64_t t = begin; t < end; ++t)
                                     {
                                         size_t pos = start_pos + static_cast<size_t>(t);
                                         int64_t added_idx = diff.added_indices[pairs_to_process + static_cast<size_t>(t)];
                                         if (has_gate_buffers)
                                             std::memcpy(active_gate_buffer.get() + pos * gate_row_size,
                                                         gate_memory_pool.get() + added_idx * gate_row_size,
                                                         gate_row_size * sizeof(float));

                                         std::memcpy(active_up_buffer.get() + pos * up_row_size,
                                                     up_memory_pool.get() + added_idx * up_row_size,
                                                     up_row_size * sizeof(float));

                                         std::memcpy(active_down_buffer.get() + pos * hidden_dim,
                                                     down_memory_pool_transposed.get() + added_idx * hidden_dim,
                                                     hidden_dim * sizeof(float));
                                     }
                                 });
            }
            else
            {
                for (size_t t = 0; t < to_append; ++t)
                {
                    size_t pos = start_pos + t;
                    int64_t added_idx = diff.added_indices[pairs_to_process + t];
                    if (has_gate_buffers)
                        std::memcpy(active_gate_buffer.get() + pos * gate_row_size,
                                    gate_memory_pool.get() + added_idx * gate_row_size,
                                    gate_row_size * sizeof(float));

                    std::memcpy(active_up_buffer.get() + pos * up_row_size,
                                up_memory_pool.get() + added_idx * up_row_size,
                                up_row_size * sizeof(float));

                    std::memcpy(active_down_buffer.get() + pos * hidden_dim,
                                down_memory_pool_transposed.get() + added_idx * hidden_dim,
                                hidden_dim * sizeof(float));
                }
            }

            // Sequential metadata updates
            active_indices.resize(start_pos + to_append);
            for (size_t t = 0; t < to_append; ++t)
            {
                size_t pos = start_pos + t;
                int64_t added_idx = diff.added_indices[pairs_to_process + t];
                active_indices[pos] = added_idx;
                index_to_position[added_idx] = pos;
            }
        }

        // Handle remaining removals (if more removals than additions)
        for (size_t i = pairs_to_process; i < num_removals; ++i)
        {
            int64_t removed_idx = diff.removed_indices[i];
            auto it = index_to_position.find(removed_idx);
            if (it != index_to_position.end())
            {
                // Check if removing this index would leave us with too few active indices
                if (active_indices.size() <= min_active_indices)
                {
                    // Skip this removal to maintain minimum cache size
                    break;
                }

                size_t pos_to_remove = it->second;
                size_t last_pos = active_indices.size() - 1;

                if (pos_to_remove != last_pos)
                {
                    // Move last element to fill gap
                    int64_t last_idx = active_indices[last_pos];

                    if (has_gate_buffers)
                        std::memcpy(active_gate_buffer.get() + pos_to_remove * gate_row_size,
                                    active_gate_buffer.get() + last_pos * gate_row_size,
                                    gate_row_size * sizeof(float));

                    std::memcpy(active_up_buffer.get() + pos_to_remove * up_row_size,
                                active_up_buffer.get() + last_pos * up_row_size,
                                up_row_size * sizeof(float));

                    std::memcpy(active_down_buffer.get() + pos_to_remove * hidden_dim,
                                active_down_buffer.get() + last_pos * hidden_dim,
                                hidden_dim * sizeof(float));

                    // Update tracking
                    active_indices[pos_to_remove] = last_idx;
                    index_to_position[last_idx] = pos_to_remove;
                }

                // Remove last element
                active_indices.pop_back();
                index_to_position.erase(it);
            }
        }

        // Rebuild tensor views using from_blob (no copying!)
        rebuild_tensor_views();
        cache_valid = true;
        current_mask = normalized_mask.clone();
    }

    // Getters remain the same
    torch::Tensor get_concat_weight() const
    {
        TORCH_CHECK(cache_valid, "Cache is not valid");
        return active_weights_cache;
    }

    torch::Tensor get_active_down_weight() const
    {
        TORCH_CHECK(cache_valid, "Cache is not valid");
        return active_downs_cache;
    }

    int64_t get_num_active() const
    {
        return static_cast<int64_t>(active_indices.size());
    }

    // Destructor - no manual cleanup needed with smart pointers
    ~WeightCache() = default;
};
