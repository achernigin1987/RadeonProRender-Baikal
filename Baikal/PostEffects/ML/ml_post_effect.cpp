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

#include "PostEffects/ML/ml_post_effect.h"
#include "denoiser_preprocessor.h"
#include "upsampler_preprocessor.h"

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif

namespace Baikal
{
    namespace PostEffects {

        using float3 = RadeonRays::float3;
        using OutputType = Renderer::OutputType;

        MLPostEffect::MLPostEffect(ModelType model_type, CLWContext context, const CLProgramManager* program_manager)
#ifdef BAIKAL_EMBED_KERNELS
        : ClwPostEffect(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers),
#else
        : ClwPostEffect(context, program_manager, "../Baikal/Kernels/CL/denoise.cl")
#endif
        , m_model_holder(nullptr)
        , m_inference(nullptr)
        , m_type(model_type)
        , m_start_seq(0)
        , m_last_seq(0)
        , m_program(program_manager)
        {
            RegisterParameter("gpu_memory_fraction", .7f);
            RegisterParameter("visible_devices", std::string());
            RegisterParameter("start_spp", 1u);
            RegisterParameter("every_frame", 0u);

            switch (m_type)
            {
            case ModelType::kDenoiser:
                m_input_data_type = InputDataType::kColorAlbedoDepthNormal9;
                break;
            case ModelType::kUpsampler:
                m_input_data_type = InputDataType::kColor3;
                break;
            default:
                throw std::logic_error("Unsupported model type");
            }
        }

        void MLPostEffect::CreateModelHolder()
        {
            std::string model_path;

            switch (m_type)
            {
            case ModelType::kDenoiser:
                if (m_input_data_type == InputDataType::kColorAlbedoDepthNormal9)
                {
                    model_path = "models/color_albedo_depth_normal_9_v3.json";
                }
                else
                {
                    throw std::logic_error("Unsupported denoiser inputs");
                }
                break;
            case ModelType::kUpsampler:
                if (m_input_data_type == InputDataType::kColor3)
                {
                    model_path = "models/esrgan-03x2x32-273866.json";
                }
                else
                {
                    throw std::logic_error("Unsupported upsampler inputs");
                }
                break;
            default:
                throw std::logic_error("Unsupported model type");
            }

            m_model_holder = std::make_unique<ModelHolder>(
                m_type,
                model_path,
                GetParameter("gpu_memory_fraction").GetFloat(),
                GetParameter("visible_devices").GetString(),
                GetContext().GetCommandQueue(0));
        }

        void MLPostEffect::CreateInference()
        {
            m_process_every_frame = static_cast<bool>(GetParameter("every_frame").GetUint());
            m_inference = std::make_unique<Inference>(
                m_model_holder.get(),
                m_input_height,
                m_input_width);
        }

        void MLPostEffect::CreatePreprocessor()
        {
            // init preprocessing
            switch (m_type)
            {
            case ModelType::kDenoiser:
                m_preprocessor = std::make_unique<DenoiserPreprocessor>(
                    m_model_holder.get(), 
                    GetContext(), 
                    m_program);
                break;
            case ModelType::kUpsampler:
                m_preprocessor = std::make_unique<UpsamplerPreprocessor>(
					m_model_holder.get(),
					GetContext(),
                    m_program);
                break;
            default:
                throw std::logic_error("unsupported model type");
            }

            m_preprocessor->SetStartSpp(GetParameter("start_spp").GetUint());
        }

        void MLPostEffect::Init(InputSet const& input_set, Output& output)
        {
            auto aov = input_set.begin()->second;

            m_input_width = aov->width();
            m_input_height = aov->height();

            CreateModelHolder();
            CreatePreprocessor();
            CreateInference();

            auto output_info = m_inference->GetOutputInfo();

            m_last_image = CLWBuffer<float3>::Create(GetContext(),
                                                     CL_MEM_READ_WRITE,
                                                     output_info.width * output_info.height);

            m_host = std::vector<float3>(output_info.width * output_info.height);
        }

        void MLPostEffect::Apply(InputSet const& input_set, Output& output)
        {
            if (m_input_width != input_set.begin()->second->width() ||
                m_input_height != input_set.begin()->second->height())
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
                throw std::runtime_error("MLPostEffect::Apply(...): can not cast output");
            }

            auto context = GetContext();
            auto shape = m_inference->GetInputInfo();
            auto input = m_preprocessor->Preprocess(input_set);

            Image res;
            if (m_process_every_frame)
            {
                m_inference->PushInput(std::move(input));
                res = m_inference->PopOutput();
            }
            else
            {
                if (input.tag == 1)
                {
                    m_start_seq = m_last_seq + 1;
                }

                if (input.image != nullptr)
                {
                    input.tag = ++m_last_seq;
                    m_inference->PushInput(std::move(input));
                }

                res = m_inference->TryPopOutput();
            }

            if (res.image != nullptr && (res.tag >= m_start_seq || m_process_every_frame))
            {
                size_t res_size;
                auto res_data = static_cast<float*>(mlMapImage(res.image, &res_size));

                if (res_data == nullptr)
                {
                    throw std::runtime_error("map input image is failed");
                }

                auto dest = m_host.data();
                auto source = res_data;
                auto output_shape = m_inference->GetOutputInfo();
                for (auto i = 0u; i < output_shape.width * output_shape.height; ++i)
                {
                    dest->x = *source++;
                    dest->y = *source++;
                    dest->z = *source++;
                    dest->w = 1;
                    ++dest;
                }

                mlUnmapImage(res.image, res_data);

                context.WriteBuffer(0,
                                    m_last_image,
                                    m_host.data(),
                                    res_size / (3 * sizeof(float)));
                // Copy postprocessed image
                context.CopyBuffer(0,
                    m_last_image,
                    clw_inference_output->data(),
                    0 /* srcOffset */,
                    0 /* destOffset */,
                    m_last_image.GetElementCount()).Wait();
            }
            else
            {
                // Postprocessed image is not ready yet.
                // Therefore, we'll output source (not-postprocessed) image
                auto color = dynamic_cast<ClwOutput*>(input_set.at(OutputType::kColor))->data();

                if (m_type == ModelType::kDenoiser)
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
                    Resize_2x(clw_inference_output->data(), color);
                }
            }
        }

        void MLPostEffect::SetParameter(std::string const& name, Param value)
        {
            auto param = GetParameter(name);
            PostEffect::SetParameter(name, value);
            m_is_dirty = true;
        }

        PostEffect::InputTypes MLPostEffect::GetInputTypes() const
        {
            return DataPreprocessor::GetInputTypes(m_input_data_type);
        }

        void MLPostEffect::Resize_2x(CLWBuffer<RadeonRays::float3> dst, CLWBuffer<RadeonRays::float3> src)
        {
            auto context = GetContext();

            if (m_resizer_cache.GetElementCount() < 2 * src.GetElementCount())
            {
                m_resizer_cache = CLWBuffer<float3>::Create(context,
                                                            CL_MEM_READ_WRITE,
                                                            2 * src.GetElementCount());
            }

            auto scale_x = GetKernel("BicubicUpscale2x_X");

            unsigned argc = 0;
            scale_x.SetArg(argc++, m_resizer_cache);
            scale_x.SetArg(argc++, src);
            scale_x.SetArg(argc++, m_input_width);
            scale_x.SetArg(argc++, m_input_height);

            // run BicubicUpScale2x_X kernel
            auto thread_num = ((2 * m_input_width * m_input_height + 63) / 64) * 64;
            context.Launch1D(0,
                             thread_num,
                             64,
                             scale_x);

            auto scale_y = GetKernel("BicubicUpscale2x_Y");

            argc = 0;
            scale_y.SetArg(argc++, dst);
            scale_y.SetArg(argc++, m_resizer_cache);
            scale_y.SetArg(argc++, 2 * m_input_width);
            scale_y.SetArg(argc++, m_input_height);

            // run BicubicUpScale2x_Y kernel
            thread_num = ((4 * m_input_width * m_input_height + 63) / 64) * 64;
            context.Launch1D(0,
                             thread_num,
                             64,
                             scale_y).Wait();
        }
    }
}
