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

#include <fstream>
#include <sstream>


namespace Baikal
{
    namespace PostEffects
    {
        using float3 =  RadeonRays::float3;
        using OutputType = Renderer::OutputType;

        std::unique_ptr<Inference> CreateMLDenoiser(MLDenoiserInputs inputs,
                                                    float gpu_memory_fraction,
                                                    std::string const& visible_devices,
                                                    std::size_t width,
                                                    std::size_t height)
        {
            std::string model_path;
            std::size_t input_channels;
            switch (inputs)
            {
            case MLDenoiserInputs::kColorDepthNormalGloss7:
                model_path = "models/color_depth_normal_gloss_7.pb";
                input_channels = 7;
                break;

            case MLDenoiserInputs::kColorAlbedoNormal8:
                model_path = "models/color_albedo_normal_8.pb";
                input_channels = 8;
                break;
            }

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
                m_layout.emplace_back(OutputType::kColor, 3);
                m_layout.emplace_back(OutputType::kDepth, 1);
                m_layout.emplace_back(OutputType::kViewShadingNormal, 2);
                m_layout.emplace_back(OutputType::kGloss, 1);
                break;

            case MLDenoiserInputs::kColorAlbedoNormal8:
                m_layout.emplace_back(OutputType::kColor, 3);
                m_layout.emplace_back(OutputType::kAlbedo, 3);
                m_layout.emplace_back(OutputType::kViewShadingNormal, 2);
                break;
            }
        }

        template <class ClType, class Type>
        void MLDenoiser::ProcessOutput(const CLWBuffer<float3>& input,
                                       Tensor::ValueType* host_mem,
                                       std::size_t channels)
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
                                        (Type*)m_host_cache.get(),
                                        input_buf.GetElementCount()).Wait();
            {
                static int frame = 0;
                std::ostringstream name;
                name << "/storage/denoise/tmp/baikal/depth_" << frame << ".bin";
                std::ofstream out(name.str(), std::ios_base::binary);
                out.write(reinterpret_cast<char*>(m_host_cache.get()),
                          input_buf.GetElementCount() * sizeof(float3));
                ++frame;
            }

            auto dest = host_mem;
            auto source = reinterpret_cast<float3*>(m_host_cache.get());
            for (auto i = 0u; i < input_buf.GetElementCount(); ++i)
            {
                *dest++ = source->x;
                dest += channels - 1;
                ++source;
            }
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

                auto clw_output = dynamic_cast<ClwOutput*>(input);
                auto device_mem = clw_output->data();

                switch (type)
                {
                case OutputType::kColor:
                {
                    m_context->ReadBuffer<float3>(0,
                                                  device_mem,
                                                  reinterpret_cast<float3*>(m_host_cache.get()),
                                                  device_mem.GetElementCount()).Wait();

                    auto dest = host_mem;
                    auto source = reinterpret_cast<float3*>(m_host_cache.get());
                    for (auto i = 0u; i < shape.width * shape.height; ++i)
                    {
                        *dest++ = source->x;
                        *dest++ = source->y;
                        *dest++ = source->z;
                        dest += shape.channels - 3;
                        ++source;
                    }
                    break;
                }
                case OutputType::kDepth:
                {
                    ProcessOutput<cl_float3, float3>(device_mem,
                                                     host_mem,
                                                     shape.channels);
//                        m_context->ReadBuffer<float3>(0,
//                                                      device_mem,
//                                                      reinterpret_cast<float3*>(m_host_cache.get()),
//                                                      device_mem.GetElementCount()).Wait();

//                        // copy only the first channel
//                        auto dest = host_mem;
//                        auto source = reinterpret_cast<float3*>(m_host_cache.get());
//                        for (auto i = 0u; i < shape.width * shape.height; ++i)
//                        {
//                            *dest++ = source->x;
//                            dest += shape.channels - 1;
//                            ++source;
//                        }
                    break;
                }
                case OutputType::kViewShadingNormal:
                {
                    m_context->ReadBuffer<float3>(0,
                                                  device_mem,
                                                  reinterpret_cast<float3*>(m_host_cache.get()),
                                                  device_mem.GetElementCount()).Wait();

                    // copy only the first two channels
                    auto dest = host_mem;
                    auto source = reinterpret_cast<float3*>(m_host_cache.get());
                    for (auto i = 0u; i < shape.width * shape.height; ++i)
                    {
                        *dest++ = source->x;
                        *dest++ = source->y;
                        dest += shape.channels - 2;
                        ++source;
                    }

                    break;
                }
                case OutputType::kGloss:
                {
                    m_context->ReadBuffer<float3>(0,
                                                  device_mem,
                                                  reinterpret_cast<float3*>(m_host_cache.get()),
                                                  device_mem.GetElementCount()).Wait();

                    // copy only the first channel
                    auto dest = host_mem;
                    auto source = reinterpret_cast<float3*>(m_host_cache.get());
                    for (auto i = 0u; i < shape.width * shape.height; ++i)
                    {
                        *dest++ = source->x;
                        dest += shape.channels - 1;
                        ++source;
                    }
                    break;
                }
                default:
                    break;
                }
                host_mem += input_desc.second;
            }

            {
                static int frame = 0;
                std::ostringstream name;
                name << "/storage/denoise/tmp/baikal/input_" << frame << ".bin";
                std::ofstream out(name.str(), std::ios_base::binary);
                out.write(reinterpret_cast<char*>(tensor.data()),
                          tensor.size() * sizeof(float));
                ++frame;
            }

            static int frames_to_start = 8;
            if (frames_to_start == 0)
            {
                m_inference->PushInput(std::move(tensor));
            }
            else
            {
                frames_to_start--;
            }

            auto clw_inference_output = dynamic_cast<ClwOutput*>(&output);

            if (!clw_inference_output)
            {
                throw std::runtime_error("MLDenoiser::Apply(...): can not cast output");
            }

            auto inference_res = m_inference->PopOutput();

            if (!inference_res.empty())
            {
                {
                    static int frame = 0;
                    std::ostringstream name;
                    name << "/storage/denoise/tmp/baikal/output_" << frame << ".bin";
                    std::ofstream out(name.str(), std::ios_base::binary);
                    out.write(reinterpret_cast<char*>(inference_res.data()),
                              inference_res.size() * sizeof(Tensor::ValueType));
                    ++frame;
                }

                auto dest = reinterpret_cast<float3*>(m_host_cache.get());
                auto source = inference_res.data();
                for (auto i = 0u; i < shape.width * shape.height; ++i)
                {
                    dest->x = *source++;
                    dest->y = *source++;
                    dest->z = *source++;
                    dest->w = 1;
                    ++dest;
                }

                m_context->WriteBuffer<float3>(0,
                    clw_inference_output->data(),
                    reinterpret_cast<float3*>(m_host_cache.get()),
                    inference_res.size() / 3).Wait();

                m_last_image = std::move(inference_res);
            }
            else if (!m_last_image.empty())
            {
                auto dest = reinterpret_cast<float3*>(m_host_cache.get());
                auto source = m_last_image.data();
                for (auto i = 0u; i < shape.width * shape.height; ++i)
                {
                    dest->x = *source++;
                    dest->y = *source++;
                    dest->z = *source++;
                    dest->w = 1;
                    ++dest;
                }

                m_context->WriteBuffer<float3>(0,
                                               clw_inference_output->data(),
                                               reinterpret_cast<float3*>(m_host_cache.get()),
                                               inference_res.size() / 3).Wait();
            }
            else
            {
                m_context->CopyBuffer<float3>(0,
                                      static_cast<ClwOutput*>(input_set.at(OutputType::kColor))->data(),
                                      clw_inference_output->data(),
                                      0,
                                      0,
                                      shape.width * shape.height).Wait();
            }
        }
    }
}
