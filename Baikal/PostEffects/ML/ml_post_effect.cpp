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

#include "ml_post_effect.h"
#include "denoiser_preprocess.h"
#include "sisr_preprocess.h"

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif

namespace Baikal
{
    namespace PostEffects {

        MlPostEffect::MlPostEffect(CLWContext context, CLProgramManager* program_manager, PostEffectType type)
#ifdef BAIKAL_EMBED_KERNELS
        : ClwPostEffect(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers),
#else
        : ClwPostEffect(context, program_manager, "../Baikal/Kernels/CL/denoise.cl")
#endif
        , m_type(type)
        , m_is_dirty(true)
        , m_program(program_manager)
        {
            m_inference = nullptr;

            RegisterParameter("gpu_memory_fraction", .7f);
            RegisterParameter("visible_devices", std::string());
        }

        Inference::Ptr MlPostEffect::CreateInference(std::uint32_t width, std::uint32_t height)
        {
            auto gpu_memory_fraction = GetParameter("gpu_memory_fraction").GetFloat();
            auto visible_devices = GetParameter("visible_devices").GetString();
            auto start_spp = GetParameter("start_spp").GetUint();

            std::string model_path;

            switch (m_type)
            {
                case PostEffectType::kDenoiser:
                    m_preproc = std::unique_ptr<DataPreprocess>(
                            new DenoiserPreprocess(
                                    GetContext(),
                                    m_program,
                                    width,
                                    height,
                                    start_spp));

                    return std::unique_ptr<Inference>(
                            new Inference("models/color_albedo_depth_normal_9_v3.pb",
                                          {ML_FLOAT32, width, height},
                                          {ML_FLOAT32, width, height},
                                          gpu_memory_fraction,
                                          visible_devices));
                case PostEffectType::kSisr:
                    m_preproc = std::unique_ptr<DataPreprocess>(
                            new SisrPreprocess(
                                    GetContext(),
                                    m_program,
                                    width,
                                    height,
                                    start_spp));

                    return std::unique_ptr<Inference>(
                            new Inference("models/esrgan-05x3x32-198135.pb",
                                          {ML_FLOAT32, width, height},
                                          {ML_FLOAT32, 2 * width, 2 * height},
                                          gpu_memory_fraction,
                                          visible_devices));

                default:
                    throw std::logic_error("Unsupported model type");
            }
        }

        void MlPostEffect::Init(InputSet const& input_set, Output& output)
        {
            auto aov = input_set.begin()->second;

            m_width = aov->width();
            m_height = aov->height();

            m_inference = CreateInference(m_width, m_height);
        }


        void MlPostEffect::Apply(InputSet const& input_set, Output& output)
        {
            if (m_width != input_set.begin()->second->width() ||
                m_height != input_set.begin()->second->height())
            {
                m_is_dirty = true;
            }

            if (m_is_dirty)
            {
                Init(input_set, output);
                m_is_dirty = false;
            }

            auto shape = m_inference->GetInputShape();

            auto image = m_preproc->MakeInput(input_set);
        }

        void MlPostEffect::SetParameter(std::string const& name, Param value)
        {
            auto param = GetParameter(name);
            PostEffect::SetParameter(name, value);
            m_is_dirty = true;
        }
    }
}