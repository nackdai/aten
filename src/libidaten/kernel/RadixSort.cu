#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "kernel/RadixSort.h"
#include "cuda/cudadefs.h"
#include "cuda/helper_math.h"
#include "cuda/cudautil.h"

namespace idaten
{
    std::vector<float> CpuRadixSort(const std::vector<float>& initialKeys)
    {
        const size_t num_keys = initialKeys.size();

        std::vector<float> key_input = initialKeys;
        std::vector<float> key_output(num_keys);
        std::vector<int32_t> prefix_sum(num_keys);

        // The loop count is the bits of the key type.
        constexpr auto loop_count = sizeof(decltype(initialKeys[0])) * 8;

        for (size_t bit_pos = 0; bit_pos < loop_count; bit_pos++)
        {
            // inclusive scan
            for (size_t i = 0; i < num_keys; i++)
            {
                // Assume non-negative valid float value.
                auto f = key_input[i];
                uint32_t key = *reinterpret_cast<uint32_t*>(&f);
                bool bit = (key >> bit_pos) & 0x01;

                // Sum the count of false bits for the current bit position.
                if (i == 0) {
                    prefix_sum[i] = !bit;
                }
                else {
                    prefix_sum[i] = prefix_sum[i - 1] + !bit;
                }
            }

            // bucket sort
            for (size_t i = 0; i < num_keys; i++)
            {
                const uint32_t si = i == 0
                    ? 0
                    : prefix_sum[i - 1];

                // prefix_sum store the accumulated count of false bits.
                // So, the total count of false bits is the last value of prefix_sum.
                const uint32_t total_false = prefix_sum.back();

                auto f = key_input[i];
                uint32_t key = *reinterpret_cast<uint32_t*>(&f);
                bool bit = (key >> bit_pos) & 0x01;

                const auto exchange_target_index = (!bit) ? si : i - si + total_false;

                key_output[exchange_target_index] = key_input[i];
            }

            std::swap(key_input, key_output);
        }

        return std::move(key_input);
    }

#define BLOCK_SIZE 1024

    __global__ void inclusive_scan(
        uint32_t* prefix_sum_output,
        const uint32_t* input,
        uint32_t* block_sum_output,
        int32_t bit_pos,
        int32_t curr_num)
    {
        // Exclusive scan:
        // https://github.com/shineyruan/CUDA-Stream-Compaction

        // Number of array is BLOCK_SIZE.
        extern __shared__ uint32_t temp[];

        // To access to the arrays locally within the threading block.
        const int32_t thread_idx = threadIdx.x;

        // To access to the array globally.
        const int32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Handle non-power-of-2 sizes by filling out-of-range with 0.
        temp[thread_idx] = (global_idx < curr_num) ? input[global_idx] : 0;
        __syncthreads();

        // Up-Sweep.
        int32_t offset = 1;
        for (int32_t d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
            if (thread_idx < d) {
                const int32_t ai = offset * (2 * thread_idx + 1) - 1;
                const int32_t bi = offset * (2 * thread_idx + 2) - 1;
                temp[bi] += temp[ai];
            }
            offset *= 2;
            __syncthreads();
        }

        // Save the block sum here before it's overwritten to 0 in Down-Sweep.
        // Block sum is stored per block. So, access index has to be the block index.
        if (thread_idx == blockDim.x - 1 && block_sum_output != nullptr) {
            block_sum_output[blockIdx.x] = temp[thread_idx];
        }

        // Set the last element to 0 for Exclusive Scan.
        if (thread_idx == 0) {
            temp[BLOCK_SIZE - 1] = 0;
        }
        __syncthreads();

        // Down-Sweep.
        for (int32_t d = 1; d < BLOCK_SIZE; d *= 2) {
            offset >>= 1;
            if (thread_idx < d) {
                const int32_t ai = offset * (2 * thread_idx + 1) - 1;
                const int32_t bi = offset * (2 * thread_idx + 2) - 1;

                const auto t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
            __syncthreads();
        }

        // Adjustment for Inclusive.
        // Add the original input to the Exclusive result.
        if (global_idx < curr_num) {
            // Inclusive result = Exclusive result + original value.
            int32_t val = (global_idx < curr_num) ? input[global_idx] : 0;
            prefix_sum_output[global_idx] = temp[thread_idx] + val;
        }
    }

    // Outputs are calculated per block.
    // To concatenate the separated outputs as one output, need to
    __global__ void add_offsets(
        uint32_t* outputs,
        const uint32_t* block_offsets,
        uint32_t num_blocks)
    {
        const int32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_idx >= num_blocks) {
            return;
        }

        // Block 0 has offset 0, so add to Block 1 onwards
        if (blockIdx.x > 0) {
            // NOTE:
            // Before this kernel, inclusive scan is done per block.
            // As the one big array, inclusive scan is not done entirely.
            // So, we need to concatenate the bocks by adding the last element of the previous block.
            // e.g.
            // | block 0    | block 1    |
            // | 1, 2, 3, 4 | 0, 2, 4, 6 |
            //
            // | block 0    | block 1     |
            // | 1, 2, 3, 4 | 4, 6, 8, 10 |
            //                - add `4` which is the last element of `block 0`.
            outputs[global_idx] += block_offsets[blockIdx.x - 1];
        }
    }

    void InclusiveScan(
        CudaMemory& prefix_sum_outputs,
        CudaMemory& key_inputs,
        size_t num_of_elements,
        int32_t bit_pos)
    {
        struct Layer {
            void* data{ nullptr };
            void* block_sums{ nullptr };
            size_t num_blocks{ 0 };
            size_t num_of_elements{ 0 };
        };
        std::vector<Layer> layers;

        size_t curr_data_num = num_of_elements;

        const uint32_t block_size = BLOCK_SIZE;
        const size_t shared_mem_size = sizeof(uint32_t) * block_size;

        void* curr_in = key_inputs.data();
        void* curr_out = prefix_sum_outputs.data();

        // Upward: Block-wise scan and sum extraction.
        while (curr_data_num > 0) {
            const auto num_blocks = (curr_data_num + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // TODO:
            void* block_sums = nullptr;
            if (num_blocks > 1) {
                checkCudaErrors(cudaMalloc((void**)&block_sums, sizeof(uint32_t)* num_blocks));
            }

            inclusive_scan << <num_blocks, block_size, shared_mem_size >> > (
                reinterpret_cast<uint32_t*>(curr_out),
                reinterpret_cast<uint32_t*>(curr_in),
                reinterpret_cast<uint32_t*>(block_sums),
                bit_pos,
                curr_data_num);

            // Save the hierarchical structure
            layers.push_back({
                curr_out,
                block_sums,
                num_blocks,
                curr_data_num
            });

            if (num_blocks == 1) {
                break;
            }

            // Move to next layer
            curr_in = block_sums;
            curr_out = block_sums;
            curr_data_num = num_blocks;
        }

        // Downward: Offset propagation.
        // From the back of layers (top) in order, add to one layer below
        for (int32_t i = layers.size() - 1; i > 0; i--) {
            const auto offset_source = layers[i].data;
            const auto target_data = layers[i - 1].data;
            const auto target_n = layers[i - 1].num_of_elements;

            const auto num_blocks = (target_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            add_offsets << <num_blocks, BLOCK_SIZE >> > (
                reinterpret_cast<uint32_t*>(target_data),
                reinterpret_cast<uint32_t*>(offset_source),
                target_n);
        }

        // Clean up.
        for (auto& layer : layers) {
            if (layer.block_sums != nullptr) {
                checkCudaErrors(cudaFree(layer.block_sums));
            }
        }
    }

    template <class T>
    __global__ void extract_bits(
        uint32_t* predicate,
        const T* input,
        int32_t bit_pos,
        int32_t num_of_elements)
    {
        const int32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_idx >= num_of_elements) {
            return;
        }

        bool bit = false;

        if constexpr (std::is_same_v<T, float>) {
            auto f = input[global_idx];
            const auto key = *reinterpret_cast<uint32_t*>(&f);
            bit = (key >> bit_pos) & 0x01;
        }
        else if constexpr (std::is_same_v<T, double>) {
            auto f = input[global_idx];
            const auto key = *reinterpret_cast<uint64_t*>(&f);
            bit = (key >> bit_pos) & 0x01;
        }
        else {
            auto key = input[global_idx];
            bit = (key >> bit_pos) & 0x01;
        }

        const uint32_t value = !bit ? 1 : 0;

        // If target bit is 0 then 1, if 1 then 0 (to pack 0 elements first)
        predicate[global_idx] = value;
    }

    template <class T>
    __global__ void scatter(
        T* output,
        const T* input,
        const uint32_t* scanned_result,
        int32_t total_zeros,
        int32_t bit_pos,
        int32_t num_of_elements)
    {
        const int32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_idx >= num_of_elements) {
            return;
        }

        const auto zeros_before = (global_idx == 0)
            ? 0
            : scanned_result[global_idx - 1];

        bool is_zero = false;

        if constexpr (std::is_same_v<T, float>) {
            auto f = input[global_idx];
            const auto key = *reinterpret_cast<uint32_t*>(&f);
            is_zero = ((key >> bit_pos) & 0x01) == 0;
        }
        else if constexpr (std::is_same_v<T, double>) {
            auto f = input[global_idx];
            const auto key = *reinterpret_cast<uint64_t*>(&f);
            is_zero = ((key >> bit_pos) & 0x01) == 0;
        }
        else {
            is_zero = ((input[global_idx] >> bit_pos) & 0x01) == 0;
        }

        const auto exchange_target_index = is_zero
            ? zeros_before
            : (global_idx - zeros_before + total_zeros);

        output[exchange_target_index] = input[global_idx];
    }

    template <class T>
    void ExecRadixSort(
        TypedCudaMemory<T>& keys,   // This is input but finally this includes sorted result. So, this is input and output.
        size_t num_of_elements)
    {
        // Radix sort:
        // https://blog.siliconstudio.co.jp/2026/03/6137/

        const auto loop_count = sizeof(T) * 8;

        TypedCudaMemory<uint32_t> predicated_values(num_of_elements);
        TypedCudaMemory<uint32_t> scan_result(num_of_elements);
        TypedCudaMemory<T> tmp(num_of_elements);

        // NOTE:
        // loop_count is always a multiple of 8 = even number.
        // So, when swapping, out is always keys.
        //     in    out
        // 1  keys   tmp
        // 2  tmp    keys
        // 3  keys   tmp
        // 4  tmp    keys <= keys is output, so sorted results are stored.

        TypedCudaMemory<T>* in = &keys;
        TypedCudaMemory<T>* out = &tmp;

        for (size_t i = 0; i < loop_count; i++) {
            const auto num_blocks = (num_of_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

            extract_bits<T> << <num_blocks, BLOCK_SIZE >> > (
                predicated_values.data(),
                in->data(),
                i,
                num_of_elements);

            InclusiveScan(
                scan_result,
                predicated_values,
                num_of_elements,
                i);

            // Get the total count of zeros (last element of scan).
            decltype(scan_result)::value_type total_zeros = 0;
            checkCudaErrors(cudaMemcpy(
                &total_zeros,
                &scan_result.data()[num_of_elements - 1],
                sizeof(decltype(total_zeros)),
                cudaMemcpyDeviceToHost));

            scatter<T> << <num_blocks, BLOCK_SIZE >> > (
                out->data(),
                in->data(),
                scan_result.data(),
                total_zeros,
                i,
                num_of_elements
            );

            // swap.
            TypedCudaMemory<T>* t = in;
            in = out;
            out = t;
        }
    }

    // For test.
#if 0
    void RadixSort::test_sort()
    {
        std::vector<float> keys = {
            3.14f, 2.71f, 1.41f, 0.577f, 1.618f,
            0.18f, 4.57f, 3.21f, 1.244f, 2.33f,
        };

        std::vector<float> sorted_keys = CpuRadixSort(keys);

        TypedCudaMemory<float> key_inputs(keys.data(), keys.size());

        ExecRadixSort(key_inputs, keys.size());

        std::vector<float> result(keys.size());
        key_inputs.readFromDeviceToHostByBytes(result.data(), sizeof(float) * result.size());
    }
#endif

    RadixSort::~RadixSort()
    {
    }

    void RadixSort::init(uint32_t num)
    {
        m_32bit = true;
    }

    void RadixSort::initWith64Bit(uint32_t num)
    {
        m_32bit = false;
    }

    template <class T>
    static void radixSort(
        uint32_t num,
        TypedCudaMemory<T>& keys,
        TypedCudaMemory<uint32_t>& values,
        std::vector<T>* resultHostKeys/*= nullptr*/,
        std::vector<uint32_t>* resultHostValues/*= nullptr*/)
    {
        thrust::sort_by_key(thrust::device, keys.data(), keys.data() + num, values.data());

        if (resultHostKeys) {
            resultHostKeys->resize(num);
            keys.readFromDeviceToHostByNum(resultHostKeys, num);
        }

        if (resultHostValues) {
            resultHostValues->resize(num);
            values.readFromDeviceToHostByNum(resultHostValues, num);
        }
    }

    void RadixSort::sort(
        const std::vector<uint32_t>& keys,
        const std::vector<uint32_t>& values,
        TypedCudaMemory<uint32_t>& refSortedKeys,
        TypedCudaMemory<uint32_t>& refSortedValues,
        std::vector<uint32_t>* resultHostKeys/*= nullptr*/,
        std::vector<uint32_t>* resultHostValues/*= nullptr*/)
    {
        AT_ASSERT(keys.size() == values.size());

        const auto num = keys.size();

        thrust::host_vector<uint32_t> hostKeys(num);
        thrust::host_vector<uint32_t> hostValues(num);

        for (size_t i = 0; i < num; i++) {
            hostKeys[i] = keys[i];
            hostValues[i] = values[i];
        }

        // copy unsorted data from host to device
        thrust::device_vector<uint32_t> deviceKeys = hostKeys;
        thrust::device_vector<uint32_t> deviceValues = hostValues;

        thrust::sort_by_key(thrust::device, deviceKeys.begin(), deviceKeys.begin() + num, deviceValues.begin());

        if (resultHostKeys) {
            thrust::host_vector<uint32_t> hostKeys = deviceKeys;
            for (size_t i = 0; i < num; i++) {
                resultHostKeys->push_back(hostKeys[i]);
            }
        }

        if (resultHostValues) {
            thrust::host_vector<uint32_t> hostValues = deviceValues;
            for (size_t i = 0; i < num; i++) {
                resultHostValues->push_back(hostValues[i]);
            }
        }
    }

    void RadixSort::sort(
        uint32_t num,
        TypedCudaMemory<uint32_t>& keys,
        TypedCudaMemory<uint32_t>& values,
        std::vector<uint32_t>* resultHostKeys/*= nullptr*/,
        std::vector<uint32_t>* resultHostValues/*= nullptr*/)
    {
        AT_ASSERT(keys.num() == values.num());
        AT_ASSERT(keys.num() <= num);

        radixSort(
            num,
            keys,
            values,
            resultHostKeys,
            resultHostValues);
    }

    void RadixSort::sortWith64Bit(
        uint32_t num,
        TypedCudaMemory<uint64_t>& keys,
        TypedCudaMemory<uint32_t>& values,
        std::vector<uint64_t>* resultHostKeys/*= nullptr*/,
        std::vector<uint32_t>* resultHostValues/*= nullptr*/)
    {
        AT_ASSERT(keys.num() == values.num());
        AT_ASSERT(keys.num() <= num);

        radixSort(
            num,
            keys,
            values,
            resultHostKeys,
            resultHostValues);
    }
}
