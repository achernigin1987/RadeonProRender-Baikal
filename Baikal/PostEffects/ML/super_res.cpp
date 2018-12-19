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

#include "super_res.h"
#include "super_res_inference.h"

#include "Output/clwoutput.h"

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif

namespace Baikal
{
    namespace PostEffects
    {
        using uint32_t = std::uint32_t;
        using float3 =  RadeonRays::float3;

        namespace
        {
            std::unique_ptr<Inference> CreateInference(
                    float gpu_memory_fraction,
                    std::string const &visible_devices,
                    std::size_t width,
                    std::size_t height)
            {
                auto model_path = "models/esrgan-05x3x32-198135.pb";

                return std::make_unique<Inference>(model_path,
                                                   gpu_memory_fraction,
                                                   visible_devices,
                                                   width,
                                                   height);
            }
        }

        ////////////////////////////////////////////////
        // SuperRes implementation
        ////////////////////////////////////////////////

        SuperRes::SuperRes(const CLWContext& context, const CLProgramManager *program_manager)
#ifdef BAIKAL_EMBED_KERNELS
        : ClwPostEffect(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers),
#else
        :ClwPostEffect(context, program_manager, "../Baikal/Kernels/CL/denoise.cl"),
#endif
         m_inference(nullptr)
        {
            m_context = std::make_unique<CLWContext>(context);

            RegisterParameter("gpu_memory_fraction", .7f);
            RegisterParameter("visible_devices", std::string());
        }

        void SuperRes::Apply(InputSet const &input_set, Output &output)
        {
            auto color_aov = input_set.begin()->second;

            auto gpu_memory_fraction = GetParameter("gpu_memory_fraction").GetFloat();
            auto visible_devices = GetParameter("visible_devices").GetString();

            if (m_inference == nullptr)
            {
                m_inference = CreateInference(gpu_memory_fraction,
                                              visible_devices,
                                              color_aov->width(),
                                              color_aov->height());
            }

            auto clw_input = dynamic_cast<ClwOutput*>(color_aov);

            if (clw_input== nullptr)
            {
                throw std::runtime_error("SuperRes::Apply(..): incorrect input");
            }

            auto device_mem = clw_input->data();
            auto tensor = m_inference->GetInputData();


            m_context->ReadBuffer<float3>(0,
                                          device_mem,
                                          reinterpret_cast<float3*>(tensor.cpu_data),
                                          device_mem.GetElementCount()).Wait();

            // push tensor in model queue
            m_inference->PushInput(std::move(tensor));
            auto clw_output = dynamic_cast<ClwOutput*>(&output);
            if (clw_output == nullptr)
            {
                throw std::runtime_error("SuperRes::Apply(..): incorrect output");
            }

            auto output_device_mem = clw_output->data();

            // get another tensor from model queue
            auto res = m_inference->PopOutput();

            if (!res.is_empty)
            {
                // if returned tensor is empty return black image
                m_context->WriteBuffer<float3>(0,
                                               output_device_mem,
                                               reinterpret_cast<float3*>(res.cpu_data),
                                               output_device_mem.GetElementCount()).Wait();
            }
            // if we get upscaled image from tensor
            // than copy it into output device buffer

        }

        PostEffect::InputTypes SuperRes::GetInputTypes() const
        {
            return std::set<Renderer::OutputType>({Renderer::OutputType::kColor});
        }
    }
}