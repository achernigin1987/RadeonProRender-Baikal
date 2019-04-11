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
#include "PostEffects/ML/ml_common.h"
#include "PostEffects/ML/inference.h"
#include "PostEffects/clw_post_effect.h"


namespace Baikal
{
    namespace PostEffects
    {
        class MLPostEffect : public ClwPostEffect
        {
        public:
            MLPostEffect(ModelType type, CLWContext context, const CLProgramManager* program_manager);

            void Apply(InputSet const& input_set, Output& output) override;

            void SetParameter(std::string const& name, Param value) override;

            InputTypes GetInputTypes() const override;

            void Resize_2x(CLWBuffer<RadeonRays::float3> dst, CLWBuffer<RadeonRays::float3> src);
        private:
            void CreateModelHolder();
            void CreatePreprocessor();
            void CreateInference();

            void Init(InputSet const& input_set, Output& output);

            ModelType m_type;
            InputDataType m_input_data_type;

            bool m_is_dirty = true;
            bool m_process_every_frame = false;

            std::vector<RadeonRays::float3> m_host;
            CLWBuffer<RadeonRays::float3> m_last_image;
            CLWBuffer<RadeonRays::float3> m_resizer_cache;

            std::unique_ptr<ModelHolder> m_model_holder;
            std::unique_ptr<DataPreprocessor> m_preprocessor;
            std::unique_ptr<Inference> m_inference;

            std::uint32_t m_input_width = 0;
            std::uint32_t m_input_height = 0;

            std::uint32_t m_start_seq = 0;
            std::uint32_t m_last_seq = 0;
            const CLProgramManager *m_program;
        };
    }
}
