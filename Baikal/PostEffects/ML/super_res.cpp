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
#include "super_res_infer_impl.h"

#include "Output/clwoutput.h"

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif

#include <queue>

namespace Baikal
{
    namespace PostEffects
    {
        using uint32_t = std::uint32_t;
        using float3 =  RadeonRays::float3;


        ////////////////////////////////////////////////
        // SuperRes implementation
        ////////////////////////////////////////////////

        SuperRes::SuperRes(const CLWContext& context, const CLProgramManager *program_manager)
#ifdef BAIKAL_EMBED_KERNELS
        : ClwPostEffect(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers),
#else
        : ClwPostEffect(context, program_manager, "../Baikal/Kernels/CL/denoise.cl"),
#endif
          m_inference(nullptr)
        {
            m_context = std::make_unique<CLWContext>(context);
        }

        void SuperRes::Apply(InputSet const &input_set, Output &output)
        {
            auto color_aov = input_set.begin()->second;

            if (m_inference == nullptr)
            {
                m_inference = std::make_unique<SuperResInferImpl>(
                        SuperResInferImpl(color_aov->width(), color_aov->height()));
            }

            auto clw_input = dynamic_cast<ClwOutput*>(color_aov);

            if (clw_input== nullptr)
            {
                throw std::runtime_error("SuperRes::Apply(..): incorrect input");
            }

            auto device_mem = clw_input->data();
            auto tensor = m_inference->GetInputTensor();


            m_context->ReadBuffer<float3>(0,
                                          device_mem,
                                          reinterpret_cast<float3*>(tensor.data()),
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

            if (res.empty())
            {
                // if returned tensor is empty return black image
                memset(res.data(), 0, output_device_mem.GetElementCount());
            }

            // if we get upscaled image from tensor
            // than copy it into output device buffer
            m_context->WriteBuffer<float3>(0,
                                           output_device_mem,
                                           reinterpret_cast<float3*>(res.data()),
                                           output_device_mem.GetElementCount());
        }

        PostEffect::InputTypes SuperRes::GetInputTypes() const
        {
            return std::set<Renderer::OutputType>({Renderer::OutputType::kColor});
        }
    }
}