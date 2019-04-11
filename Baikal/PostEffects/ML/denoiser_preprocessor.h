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

#include "data_preprocessor.h"
#include "PostEffects/ML/model_holder.h"
#include "CLW.h"
#include "Utils/clw_class.h"

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif

#include <cstddef>
#include <memory>
#include <vector>


namespace Baikal
{
    namespace PostEffects
    {
        class DenoiserPreprocessor: public DataPreprocessor
        {
        public:
            DenoiserPreprocessor(ModelHolder* model_holder,
                                 CLWContext clcontext,
                                 CLProgramManager const* program_manager,
                                 std::uint32_t start_spp = 8);

            Image Preprocess(PostEffect::InputSet const& inputs) override;
            Image Preprocess(PostEffect::InputSet const& inputs, bool use_interop);

            std::tuple<std::uint32_t, std::uint32_t> ChannelsNum() const override;
        private:
            void Init(std::uint32_t width, std::uint32_t height);
            Image DoPreprocess(PostEffect::InputSet const& inputs);
            //Image CopyToHost(Image& image);

            // layout of the outputs in input tensor in terms of channels
            using MemoryLayout = std::vector<std::pair<Renderer::OutputType, int>>;

            void DivideBySampleCount(CLWBuffer<RadeonRays::float3> const& dst,
                                     CLWBuffer<RadeonRays::float3> const& src);

            ModelHolder* m_model_holder;
            InputDataType m_model;
            CLWParallelPrimitives m_primitives;
            std::uint32_t m_width, m_height;
            std::uint32_t m_channels = 0;

            MemoryLayout m_layout;
            CLWBuffer<float> m_cache;
            CLWBuffer<float> m_input;
            ml_image m_image;
            bool m_is_initialized = false;
        };
    }
}
