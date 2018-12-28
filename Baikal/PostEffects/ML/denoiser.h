/**********************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#include "PostEffects/ML/ml_post_effect.h"
#include "PostEffects/ML/inference.h"

#include "PostEffects/post_effect.h"
#include "PostEffects/clw_post_effect.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class CLWContext;
class CLWParallelPrimitives;

template <class T>
class CLWBuffer;

namespace Baikal
{
    namespace PostEffects
    {
        enum class MLDenoiserInputs
        {
            kColorDepthNormalGloss7,
            kColorAlbedoNormal8,
            kColorAlbedoDepthNormal9
        };

        class MLDenoiser : public MlPostEffect
        {
        public:

            MLDenoiser(const CLWContext& context, const CLProgramManager *program_manager);

            InputTypes GetInputTypes() const override;

        private:
            bool PrepeareInput(BufferPtr device_buffer, InputSet const& input_set) override;
            void PrepeareOutput(Image const& inference_res, Output& output) override;

            using MemoryLayout = std::vector<std::pair<Renderer::OutputType, std::size_t>>;

            void DivideBySampleCount(CLWBuffer<RadeonRays::float3> dst,
                                       CLWBuffer<RadeonRays::float3> src);

            void WriteToInputs(CLWBuffer<RadeonRays::float3> dst_buffer,
                               CLWBuffer<RadeonRays::float3> src_buffer,
                               int dst_channels_offset,
                               int src_channels_offset,
                               int src_channels_num,
                               int channels_to_copy);

            MLDenoiserInputs m_inputs;
            MemoryLayout m_layout;
            std::unique_ptr<CLWParallelPrimitives> m_primitives;
            // GPU cache
            std::unique_ptr<CLWBuffer<float>> m_inputs_cache;
            std::unique_ptr<CLWBuffer<RadeonRays::float3>> m_device_cache;
            // CPU cache
            std::vector<RadeonRays::float3> m_host_cache;
            std::unique_ptr<CLWBuffer<RadeonRays::float3>> m_last_denoised_image;
            bool m_has_denoised_image = false;
        };
    }
}
