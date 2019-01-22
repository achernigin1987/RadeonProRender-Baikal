/**********************************************************************
 Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ********************************************************************/

#pragma once

#include "PostEffects/clw_post_effect.h"

#include "CLWBuffer.h"
#include "CLWParallelPrimitives.h"

#include <RadeonImageFilters.h>


namespace Baikal {
namespace PostEffects {

class RIFDenoiser : public ClwPostEffect
{
public:
    using Handle = std::unique_ptr<void, rif_int (*)(void*)>;

    RIFDenoiser(const CLWContext& context, const CLProgramManager* program_manager);

    InputTypes GetInputTypes() const override;

    void Apply(InputSet const& input_set, Output& output) override;

    void SetParameter(std::string const& name, Param value) override;

private:
    struct Image
    {
        CLWBuffer<float> cl;
        Handle rif;
    };

    Image CreateImage(int image_channels);

    void InitInference();

    unsigned ReadSampleCount(const CLWBuffer<RadeonRays::float3>& buffer);

    void WriteToInputs(const CLWBuffer<float>& dst_buffer,
                       const CLWBuffer<RadeonRays::float3>& src_buffer,
                       int dst_image_channels,
                       int src_image_channels,
                       int channels_to_copy);

    // OpenCL
    CLWContext m_context;
    CLWParallelPrimitives m_primitives;

    // RIF
    Handle m_rif_context;
    Handle m_rif_cmd_queue;
    Handle m_rif_image_filter;

    // GPU cache
    CLWBuffer<RadeonRays::float3> m_device_cache;
    std::vector<Image> m_inputs;
    Image m_output;

    std::uint32_t m_start_seq_num = 0;
    std::uint32_t m_last_seq_num = 0;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
    bool m_is_dirty = true;
};

} // namespace PostEffects
} // namespace Baikal
