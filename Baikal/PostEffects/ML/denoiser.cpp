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


#include "PostEffects/ML/denoiser.h"
#include "PostEffects/ML/inference_impl.h"

#include "CLWContext.h"
#include "CLWParallelPrimitives.h"
#include "Output/clwoutput.h"

#include <CLWBuffer.h>

namespace Baikal
{
    namespace PostEffects
    {
        std::unique_ptr<Inference> CreateMLDenoiser(MLDenoiserInputs inputs,
                                                    float gpu_memory_fraction,
                                                    std::string const& visible_devices,
                                                    std::size_t width,
                                                    std::size_t height)
        {
            std::string model_path = "model/path.pb";
            std::size_t input_channels = 7;
            // TODO: select model_path based on MLDenoiserInputs
            return std::make_unique<InferenceImpl>(model_path,
                                                   gpu_memory_fraction,
                                                   visible_devices,
                                                   width,
                                                   height,
                                                   input_channels);
        }

        MLDenoiser::MLDenoiser(CLWContext context, Inference::Ptr inference)
                   : m_inference(std::move(inference))
        {
            m_context = std::move(std::make_unique<CLWContext>(context));
            m_primitives = std::move(std::make_unique<CLWParallelPrimitives>(context));

            auto shape = m_inference->GetInputShape();

            size_t elems_count = sizeof(Tensor::ValueType) *
                                 std::get<0>(shape) *
                                 std::get<1>(shape);

            m_cache = std::move(std::make_unique<CLWBuffer<char>>(
                                CLWBuffer<char>::Create(*m_context,
                                                        CL_MEM_READ_WRITE,
                                                        elems_count)));

        }

        template <class ClType, class Type>
        void MLDenoiser::ProcessOutput(CLWBuffer<RadeonRays::float3> input,
                                       Tensor::ValueType* host_mem)
        {
            auto input_buf = CLWBuffer<ClType>::CreateFromClBuffer(input);
            auto normalized_buf = CLWBuffer<ClType>::CreateFromClBuffer(*m_cache);

            m_primitives->Normalize(0,
                                    input_buf,
                                    normalized_buf,
                                    (int)input_buf.GetElementCount()).Wait();

            m_context->ReadBuffer<Type>(0,
                                        CLWBuffer<Type>::CreateFromClBuffer(
                                        normalized_buf),
                                        (Type*)(host_mem),
                                        input_buf.GetElementCount()).Wait();
        }

        void MLDenoiser::Apply(InputSet const& input_set, Output& output)
        {
            size_t offset = 0;
            auto shape = m_inference->GetInputShape();
            auto width = std::get<0>(shape);
            auto height = std::get<1>(shape);

            auto tensor = m_inference->GetInputTensor();

            for (const auto& input : input_set)
            {
                auto type = input.first;
                auto clw_output = static_cast<ClwOutput*>(input.second);
                auto host_mem = tensor.data();
                auto device_mem = clw_output->data();
                
                switch (type)
                {
                    case Renderer::OutputType::kColor:
                    {
                        ProcessOutput<cl_float3, RadeonRays::float3>(device_mem, host_mem);
                        offset += 3 * width * height;
                        break;
                    }
                    case Renderer::OutputType::kDepth:
                    {
                        ProcessOutput<cl_float, Tensor::ValueType>(device_mem, host_mem + offset);
                        offset += width * height;
                        break;
                    }
                    case Renderer::OutputType::kWorldShadingNormal:
                    {
                        auto normals_place = host_mem + offset;
                        ProcessOutput<cl_float3, RadeonRays::float3>(device_mem, normals_place);

                        // remove third chanel
                        for (auto i = 0u; i < 3 * width * height; i += 3)
                        {
                            normals_place[i / 3] = normals_place[i];
                            normals_place[i / 3 + 1] = normals_place[i + 1];
                        }

                        offset += 2 * width * height;
                        break;
                    }
                    case Renderer::OutputType::kGloss:
                    {
                        auto gloss_place = host_mem + offset;
                        ProcessOutput<cl_float3, RadeonRays::float3>(device_mem, gloss_place);

                        // remove third chanel
                        for (auto i = 0u; i < 3 * width * height; i += 3)
                        {
                            gloss_place[i / 3] = gloss_place[i];
                        }
                        break;
                    }
                }
            }

            m_inference->PushInput(std::move(tensor));
            auto inference_res = m_inference->PopOutput();

            auto clw_output = dynamic_cast<ClwOutput*>(&output);

            if (!clw_output)
            {
                throw std::runtime_error("MLDenoiser::Apply(...): can not cast output");
            }

            m_context->WriteBuffer<RadeonRays::float3>(0,
                                                       clw_output->data(),
                                                       (RadeonRays::float3*)tensor.data(),
                                                       tensor.size()).Wait();
        }

    }
}
