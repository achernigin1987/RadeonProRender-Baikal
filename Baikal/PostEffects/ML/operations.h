#pragma once

#include "CLWBuffer.h"

#include <type_traits>


namespace Baikal {
namespace PostEffects {

template<class T, class U>
CLWBuffer<T> Cast(const CLWBuffer<U>& buffer)
{
    static_assert (!std::is_same<T, U>::value, "Cast is not required here");
    return CLWBuffer<T>::CreateFromClBuffer(buffer);
}

inline void DivideBySampleCount(ClwClass* clw,
                                const CLWBuffer<RadeonRays::float3>& dst_buffer,
                                const CLWBuffer<RadeonRays::float3>& src_buffer)
{
    assert(dst_buffer.GetElementCount() >= src_buffer.GetElementCount());

    auto division_kernel = clw->GetKernel("DivideBySampleCount");

    unsigned argc = 0;
    division_kernel.SetArg(argc++, dst_buffer);
    division_kernel.SetArg(argc++, src_buffer);
    division_kernel.SetArg(argc++, static_cast<cl_int>(src_buffer.GetElementCount()));

    const auto thread_num = ((src_buffer.GetElementCount() + 63) / 64) * 64;
    clw->GetContext().Launch1D(0, thread_num, 64, division_kernel);
}

inline void CopyInterleaved(ClwClass* clw,
                            const CLWBuffer<float>& dst_buffer,
                            const CLWBuffer<float>& src_buffer,
                            int image_width,
                            int image_height,
                            int dst_image_channels,
                            int dst_channels_offset,
                            int src_image_channels,
                            int src_channels_offset,
                            int channels_num_to_copy)
{
    auto copy_kernel = clw->GetKernel("CopyInterleaved");

    unsigned argc = 0;
    copy_kernel.SetArg(argc++, dst_buffer);
    copy_kernel.SetArg(argc++, src_buffer);
    copy_kernel.SetArg(argc++, image_width);
    copy_kernel.SetArg(argc++, image_height);
    copy_kernel.SetArg(argc++, dst_channels_offset);
    copy_kernel.SetArg(argc++, dst_image_channels);
    copy_kernel.SetArg(argc++, image_width);
    copy_kernel.SetArg(argc++, image_height);
    copy_kernel.SetArg(argc++, src_channels_offset);
    copy_kernel.SetArg(argc++, src_image_channels);
    copy_kernel.SetArg(argc++, channels_num_to_copy);

    // run copy_kernel
    const auto thread_num = ((image_width * image_height + 63) / 64) * 64;
    clw->GetContext().Launch1D(0, thread_num, 64, copy_kernel);
}

template<class T>
std::vector<float> ReadBuffer(const CLWContext& context, const CLWBuffer<T>& buffer)
{
    std::vector<float> data(buffer.GetElementCount() * (sizeof(T) / sizeof(float)));
    context.ReadBuffer(0 /*idx*/, buffer, reinterpret_cast<T*>(data.data()), buffer.GetElementCount()).Wait();
    return data;
}

} // namespace PostEffects
} // namespace Baikal
