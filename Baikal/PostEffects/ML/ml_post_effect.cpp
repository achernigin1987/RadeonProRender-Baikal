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

        using float3 = RadeonRays::float3;
        using OutputType = Renderer::OutputType;

        MlPostEffect::MlPostEffect(CLWContext context, const CLProgramManager* program_manager, ModelType type)
#ifdef BAIKAL_EMBED_KERNELS
        : ClwPostEffect(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers),
#else
        : ClwPostEffect(context, program_manager, "../Baikal/Kernels/CL/denoise.cl")
#endif
        , m_type(type)
        , m_is_dirty(true)
        , m_has_denoised_img(false)
        , m_start_seq(0)
        , m_last_seq(0)
        , m_program(program_manager)
        {
            m_inference = nullptr;

            RegisterParameter("gpu_memory_fraction", .7f);
            RegisterParameter("visible_devices", std::string());
            RegisterParameter("start_spp", 1u);

            // init preprocessing
            switch (m_type)
            {
                case ModelType ::kDenoiser:
                    m_preproc = std::unique_ptr<DataPreprocess>(
                            new DenoiserPreprocess(GetContext(), m_program));
                    break;
                case ModelType ::kSisr:
                    m_preproc = std::unique_ptr<SisrPreprocess>(
                            new SisrPreprocess(GetContext(), m_program));
                    break;
                default:
                    throw std::logic_error("unsupported model type");
            }
        }

        Inference::Ptr MlPostEffect::CreateInference(std::uint32_t width, std::uint32_t height)
        {
            auto gpu_memory_fraction = GetParameter("gpu_memory_fraction").GetFloat();
            auto visible_devices = GetParameter("visible_devices").GetString();
            m_preproc->ResetSpp(GetParameter("start_spp").GetUint());

            std::string model_path;

            switch (m_type)
            {
                case ModelType::kDenoiser:
                    return std::unique_ptr<Inference>(
                            new Inference("models/color_albedo_depth_normal_9_v3.pb",
                                          {ML_FLOAT32, width, height},
                                          {ML_FLOAT32, width, height},
                                          gpu_memory_fraction,
                                          visible_devices));
                case ModelType::kSisr:
                    return std::unique_ptr<Inference>(
                            new Inference("models/esrgan-05x3x32-198135.pb",
                                          {ML_FLOAT32, width, height, 3},
                                          {ML_FLOAT32, 2 * width, 2 * height, 3},
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
            auto out_shape = m_inference->GetOutputShape();

            m_last_image = CLWBuffer<float3>::Create(GetContext(),
                                                     CL_MEM_READ_WRITE,
                                                     out_shape.width * out_shape.height);

            m_host = std::vector<float3>(out_shape.width * out_shape.height);
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

            auto clw_inference_output = dynamic_cast<ClwOutput*>(&output);

            if (!clw_inference_output)
            {
                throw std::runtime_error("MlPostEffect::Apply(...): can not cast output");
            }

            auto context = GetContext();
            auto shape = m_inference->GetInputShape();
            auto input = m_preproc->MakeInput(input_set);

            // if there's no suitable image for network input
            if (input.image == nullptr)
            {
                auto color = dynamic_cast<ClwOutput*>(input_set.at(OutputType::kColor))->data();
                if (m_type == ModelType ::kDenoiser)
                {
                    context.CopyBuffer<float3>(0,
                                               color,
                                               clw_inference_output->data(),
                                               0 /* srcOffset */,
                                               0 /* destOffset */,
                                               shape.width * shape.height).Wait();
                }
                else
                {
                    Resize_x2(clw_inference_output->data(), color);
                }

                if (input.tag == 1)
                {
                    m_start_seq = m_last_seq + 1;
                    m_has_denoised_img = false;
                }
                return;
            }

            input.tag = ++m_last_seq;
            m_inference->PushInput(std::move(input));

            auto res = m_inference->PopOutput();

            if (res.image != ML_INVALID_HANDLE && res.tag >= m_start_seq)
            {
                size_t res_size;
                auto res_data = static_cast<float *>(mlMapImage(res.image, &res_size));

                if (res_data == nullptr)
                {
                    throw std::runtime_error("map input image is failed");
                }

                auto dest = m_host.data();
                auto source = res_data;
                for (auto i = 0u; i < shape.width * shape.height; ++i)
                {
                    dest->x = *source++;
                    dest->y = *source++;
                    dest->z = *source++;
                    dest->w = 1;
                    ++dest;
                }

                mlUnmapImage(res.image, res_data);

                context.WriteBuffer<float3>(0,
                                            m_last_image,
                                            m_host.data(),
                                            res_size / (3 * sizeof(float)));

                m_has_denoised_img = true;
            }

            if (m_has_denoised_img)
            {
                context.CopyBuffer<float3>(0,
                                           m_last_image,
                                           clw_inference_output->data(),
                                           0 /* srcOffset */,
                                           0 /* destOffset */,
                                           m_last_image.GetElementCount()).Wait();
            }
        }

        void MlPostEffect::SetParameter(std::string const& name, Param value)
        {
            auto param = GetParameter(name);
            PostEffect::SetParameter(name, value);
            m_is_dirty = true;
        }

        PostEffect::InputTypes MlPostEffect::GetInputTypes() const
        {
            return m_preproc->GetInputTypes();
        }
        void MlPostEffect::Resize_x2(CLWBuffer<RadeonRays::float3> dst, CLWBuffer<RadeonRays::float3> src)
        {
            auto context = GetContext();

            if (m_resizer_cache.GetElementCount() < 2 * src.GetElementCount())
            {
                m_resizer_cache = CLWBuffer<float3>::Create(context,
                                                            CL_MEM_READ_WRITE,
                                                            2 * src.GetElementCount());
            }

            auto scale_x = GetKernel("BicubicUpScaleX_x2");

            int argc = 0;
            scale_x.SetArg(argc++, m_resizer_cache);
            scale_x.SetArg(argc++, src);
            scale_x.SetArg(argc++, m_width);
            scale_x.SetArg(argc++, m_height);

            // run BicubicUpScaleX_x2 kernel
            auto thread_num = ((2 * m_width * m_height + 63) / 64) * 64;
            context.Launch1D(0,
                             thread_num,
                             64,
                             scale_x);

            auto scale_y = GetKernel("BicubicUpScaleY_x2");

            argc = 0;
            scale_y.SetArg(argc++, dst);
            scale_y.SetArg(argc++, m_resizer_cache);
            scale_y.SetArg(argc++, 2 * m_width);
            scale_y.SetArg(argc++, m_height);

            // run BicubicUpScaleY_x2 kernel
            thread_num = ((4 * m_width * m_height + 63) / 64) * 64;
            context.Launch1D(0,
                             thread_num,
                             64,
                             scale_y).Wait();
        }
    }
}