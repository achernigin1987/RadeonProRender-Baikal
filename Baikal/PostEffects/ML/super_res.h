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

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif

#include "inference.h"
#include "PostEffects/clw_post_effect.h"

template <class T>
class CLWBuffer;

namespace Baikal
{
    namespace PostEffects
    {
        class SuperRes : public ClwPostEffect
        {
        public:

            SuperRes(const CLWContext& context, const CLProgramManager *program_manager);

            void Apply(InputSet const &input_set, Output &output) override;

            InputTypes GetInputTypes() const override;

        private:
            void Tonemap(CLWBuffer<RadeonRays::float3> dst,
                         CLWBuffer<RadeonRays::float3> src);

            std::unique_ptr<CLWContext> m_context;
            Inference::Ptr m_inference;
            std::uint32_t m_start_seq_num = 0;
            std::uint32_t m_last_seq_num = 0;
            std::uint32_t m_width = 0;
            std::uint32_t m_height = 0;
            bool m_has_denoised_image;
            std::unique_ptr<CLWBuffer<RadeonRays::float3>> m_device_cache;
            std::unique_ptr<CLWBuffer<RadeonRays::float3>> m_resizer_cache;
            std::unique_ptr<CLWBuffer<float>> m_input_cache;
            std::vector<float> m_cache;
            std::unique_ptr<CLWBuffer<RadeonRays::float3>> m_last_denoised_image;
        };
    }
}