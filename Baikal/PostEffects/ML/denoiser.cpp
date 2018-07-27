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
        using OutputType = Renderer::OutputType;

        std::unique_ptr<Inference> CreateMLDenoiser(MLDenoiserInputs inputs,
                                                    float gpu_memory_fraction,
                                                    std::string const& visible_devices,
                                                    std::size_t width,
                                                    std::size_t height)
        {
            std::string model_path = "models/color_depth_normal_gloss_7.pb";
            std::size_t input_channels = 7;
            // TODO: select model_path based on MLDenoiserInputs
            return std::make_unique<InferenceImpl>(model_path,
                                                   gpu_memory_fraction,
                                                   visible_devices,
                                                   width,
                                                   height,
                                                   input_channels);
        }

        MLDenoiser::MLDenoiser(const CLWContext& context, Inference::Ptr inference, MLDenoiserInputs inputs)
                   : m_inference(std::move(inference))
        {
            m_context = std::make_unique<CLWContext>(context);
            m_primitives = std::make_unique<CLWParallelPrimitives>(context);

            auto shape = m_inference->GetInputShape();

            size_t elems_count = sizeof(Tensor::ValueType) * shape.width * shape.height * shape.channels;

            m_device_cache = std::make_unique<CLWBuffer<char>>(
                CLWBuffer<char>::Create(*m_context, CL_MEM_READ_WRITE, elems_count));

            m_host_cache = std::make_unique<std::uint8_t[]>(elems_count);

            // compute memory layout
            switch (inputs)
            {
                case MLDenoiserInputs::kColorDepthNormalGloss7:
                {
                    m_layout.emplace_back(OutputType::kColor, 3 * shape.width * shape.height);
                    m_layout.emplace_back(OutputType::kDepth, shape.width * shape.height);
                    m_layout.emplace_back(OutputType::kViewShadingNormal, 2 * shape.width * shape.height);
                    m_layout.emplace_back(OutputType::kGloss, shape.width * shape.height);
                }
            }
        }

        template <class ClType, class Type>
        void MLDenoiser::ProcessOutput(const CLWBuffer<RadeonRays::float3>& input,
                                       Tensor::ValueType* host_mem)
        {
            auto input_buf = CLWBuffer<ClType>::CreateFromClBuffer(input);
            auto normalized_buf = CLWBuffer<ClType>::CreateFromClBuffer(*m_device_cache);

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
            auto shape = m_inference->GetInputShape();

            auto tensor = m_inference->GetInputTensor();
            auto host_mem = tensor.data();

            for (const auto& input_desc : m_layout)
            {
                auto type = input_desc.first;
                auto input = input_set.at(type);

                auto clw_output = static_cast<ClwOutput*>(input);
                auto device_mem = clw_output->data();

                switch (type)
                {
                    case Renderer::OutputType::kColor:
                    {
                        ProcessOutput<cl_float3, RadeonRays::float3>(device_mem, host_mem);
                        break;
                    }
                    case Renderer::OutputType::kDepth:
                    {
                        ProcessOutput<cl_float, Tensor::ValueType>(device_mem, host_mem);
                        break;
                    }
                    case Renderer::OutputType::kViewShadingNormal:
                    {
                        ProcessOutput<cl_float3, RadeonRays::float3>(device_mem,
                            reinterpret_cast<Tensor::ValueType*>(m_host_cache.get()));

                        // copy only the first two channels
                        for (auto i = 0u; i < 3 * shape.width * shape.height; i += 3)
                        {
                            host_mem[i / 3] = m_host_cache[i];
                            host_mem[i / 3 + 1] = m_host_cache[i + 1];
                        }

                        break;
                    }
                    case Renderer::OutputType::kGloss:
                    {
                        ProcessOutput<cl_float3, RadeonRays::float3>(device_mem,
                            reinterpret_cast<Tensor::ValueType*>(m_host_cache.get()));

                        // copy only the first channel
                        for (auto i = 0u; i < 3 * shape.width * shape.height; i += 3)
                        {
                            host_mem[i / 3] = m_host_cache[i];
                        }
                        break;
                    }
                    default: break;
                }
                host_mem += input_desc.second;
            }

            m_inference->PushInput(std::move(tensor));
            auto clw_inference_output = dynamic_cast<ClwOutput*>(&output);

            if (!clw_inference_output)
            {
                throw std::runtime_error("MLDenoiser::Apply(...): can not cast output");
            }

            auto inference_res = m_inference->PopOutput();

            if (!inference_res.empty())
            {
                m_context->WriteBuffer<RadeonRays::float3>(0,
                    clw_inference_output->data(),
                    (RadeonRays::float3*)inference_res.data(),
                    inference_res.size() / inference_res.shape().channels).Wait();
            }
            else
            {
                auto shape = m_inference->GetInputShape();

                m_context->CopyBuffer<RadeonRays::float3>(0,
                                      static_cast<ClwOutput*>(input_set.at(Renderer::OutputType::kColor))->data(),
                                      clw_inference_output->data(),
                                      0,
                                      0,
                                      shape.width * shape.height).Wait();
            }
        }
    }
}
