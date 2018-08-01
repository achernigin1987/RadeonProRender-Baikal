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

#include "PostEffects/ML/inference.h"
#include "PostEffects/ML/tensor.h"
#include "PostEffects/post_effect.h"

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
        };

        class MLDenoiser : public PostEffect
        {
        public:

            MLDenoiser(const CLWContext& context, std::size_t width, std::size_t height);

            InputTypes GetInputTypes() const override;

            void Apply(InputSet const& input_set, Output& output) override;

        private:
            using MemoryLayout = std::vector<std::pair<Renderer::OutputType, std::size_t>>;

            template <class ClType, class Type>
            void ProcessOutput(const CLWBuffer<RadeonRays::float3>& input,
                               Tensor::ValueType* host_mem,
                               std::size_t channels);

            template <class T>
            T* HostCache() const
            {
                return reinterpret_cast<T*>(m_host_cache.get());
            }

            MLDenoiserInputs m_inputs;
            Inference::Ptr m_inference;
            MemoryLayout m_layout;
            std::unique_ptr<CLWContext> m_context;
            std::unique_ptr<CLWParallelPrimitives> m_primitives;
            // GPU cache
            std::unique_ptr<CLWBuffer<char>> m_device_cache;
            // CPU cache
            std::unique_ptr<std::uint8_t[]> m_host_cache;
            Tensor m_last_image;
            std::uint32_t m_start_seq_num = 0;
            std::uint32_t m_last_seq_num = 0;
        };
    }
}
