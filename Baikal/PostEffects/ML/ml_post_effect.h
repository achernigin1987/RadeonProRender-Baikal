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

#include "data_preprocess.h"
#include "PostEffects/ML/inference.h"
#include "PostEffects/clw_post_effect.h"

namespace Baikal
{
    namespace PostEffects
    {

        enum class PostEffectType
        {
            kDenoiser = 0,
            kSisr
        };

        class MlPostEffect : public ClwPostEffect
        {
        public:
            MlPostEffect(CLWContext context, CLProgramManager* program_manager, PostEffectType type);

            void Apply(InputSet const& input_set, Output& output) override;

            void SetParameter(std::string const& name, Param value) override;

        protected:
            virtual bool PrepeareInput(BufferPtr device_buffer, InputSet const& input_set) = 0;
            virtual void PrepeareOutput(Image const& inference_res, Output& output) = 0;

        private:
            Inference::Ptr CreateInference(std::uint32_t width, std::uint32_t height);
            void Init(InputSet const& input_set, Output& output);

            Inference::Ptr m_inference;
            PostEffectType m_type;
            bool m_is_dirty;
            CLWBuffer<RadeonRays::float3> m_device_buf;
            std::unique_ptr<DataPreprocess> m_preproc;
            std::uint32_t m_width, m_height;
            std::uint32_t m_start_seq, m_last_seq;
            CLProgramManager *m_program;
        };
    }
}