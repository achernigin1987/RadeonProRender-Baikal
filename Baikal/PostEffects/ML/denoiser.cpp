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

// E.g. #define ML_DENOISER_IMAGES_DIR "/images/dir"
#ifdef ML_DENOISER_IMAGES_DIR
#include <fstream>
#include <sstream>

namespace
{
    void SaveImage(char const* name, float const* buffer, std::size_t size, unsigned index)
    {
        std::ostringstream path;
        path << ML_DENOISER_IMAGES_DIR << "/" << name << "_" << index << ".bin";
        std::ofstream out(path.str(), std::ios_base::binary);
        out.write(reinterpret_cast<char const*>(buffer), size * sizeof(float));
        std::cerr << "Written: " << path.str() << "\n";
    }
}
#endif

namespace Baikal
{
    namespace PostEffects
    {
        using float3 =  RadeonRays::float3;
        using OutputType = Renderer::OutputType;

        namespace
        {
            std::unique_ptr<Inference> CreateInference(MLDenoiserInputs inputs,
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
        }

        MLDenoiser::MLDenoiser(const CLWContext& context,
                               MLDenoiserInputs inputs,
                               float gpu_memory_fraction,
                               std::string const& visible_devices,
                               std::size_t width,
                               std::size_t height)
        : m_inference(CreateInference(inputs,
                                      gpu_memory_fraction,
                                      visible_devices,
                                      width,
                                      height))
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
            auto dest = host_mem;
            auto source = HostCache<float3>();
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

            unsigned sample_count = 0;

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
                                                  HostCache<float3>(),
                                                  device_mem.GetElementCount()).Wait();

                    auto dest = host_mem;
                    auto source = HostCache<float3>();
                    sample_count = static_cast<unsigned int>(source->w);
                    (void) sample_count;
                    for (auto i = 0u; i < shape.width * shape.height; ++i)
                    {
                        dest[0] = source->x / source->w;
                        dest[1] = source->y / source->w;
                        dest[2] = source->z / source->w;
                        dest += shape.channels;
                        ++source;
                    }
                    break;
                }
                case OutputType::kDepth:
                {
                    ProcessOutput<cl_float3, float3>(device_mem,
                                                     host_mem,
                                                     shape.channels);
                    break;
                }
                case OutputType::kViewShadingNormal:
                {
                    m_context->ReadBuffer<float3>(0,
                                                  device_mem,
                                                  HostCache<float3>(),
                                                  device_mem.GetElementCount()).Wait();

                    // copy only the first two channels
                    auto dest = host_mem;
                    auto source = HostCache<float3>();
                    for (auto i = 0u; i < shape.width * shape.height; ++i)
                    {
                        if (source->w)
                        {
                            dest[0] = source->x / source->w;
                            dest[1] = source->y / source->w;
                        }
                        else
                        {
                            dest[0] = 0;
                            dest[1] = 0;
                        }
                        dest += shape.channels;
                        ++source;
                    }

                    break;
                }
                case OutputType::kGloss:
                {
                    m_context->ReadBuffer<float3>(0,
                                                  device_mem,
                                                  HostCache<float3>(),
                                                  device_mem.GetElementCount()).Wait();

                    // copy only the first channel
                    auto dest = host_mem;
                    auto source = HostCache<float3>();
                    for (auto i = 0u; i < shape.width * shape.height; ++i)
                    {
                        if (source->w)
                        {
                            dest[0] = source->x / source->w;
                        }
                        else
                        {
                            dest[0] = 0;
                        }
                        dest += shape.channels;
                        ++source;
                    }
                    break;
                }
                case OutputType::kAlbedo:
                {
                    m_context->ReadBuffer<float3>(0,
                                                  device_mem,
                                                  HostCache<float3>(),
                                                  device_mem.GetElementCount()).Wait();

                    // copy only the first channel
                    auto dest = host_mem;
                    auto source = HostCache<float3>();
                    for (auto i = 0u; i < shape.width * shape.height; ++i)
                    {
                        if (source->w)
                        {
                            dest[0] = source->x / source->w;
                            dest[1] = source->y / source->w;
                            dest[2] = source->z / source->w;
                        }
                        else
                        {
                            dest[0] = 0;
                            dest[1] = 0;
                            dest[2] = 0;
                        }
                        dest += shape.channels;
                        ++source;
                    }
                    break;
                }
                default:
                    break;
                }
                host_mem += input_desc.second;
            }

#ifdef ML_DENOISER_IMAGES_DIR
            static unsigned input_index = 0;
            SaveImage("input", tensor.data(), tensor.size(), input_index++);
#endif
            if (sample_count >= 8)
            {
                tensor.tag = ++m_last_seq_num;
                m_inference->PushInput(std::move(tensor));
            }
            else
            {
                m_start_seq_num = m_last_seq_num + 1;
                m_last_image = {};
            }

            auto clw_inference_output = dynamic_cast<ClwOutput*>(&output);

            if (!clw_inference_output)
            {
                throw std::runtime_error("MLDenoiser::Apply(...): can not cast output");
            }

            auto inference_res = m_inference->PopOutput();

            if (!inference_res.empty() && inference_res.tag >= m_start_seq_num)
            {
#ifdef ML_DENOISER_IMAGES_DIR
                SaveImage("output", inference_res.data(), inference_res.size(), inference_res.tag);
#endif
                auto dest = HostCache<float3>();
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
                    HostCache<float3>(),
                    inference_res.size() / 3).Wait();

                m_last_image = std::move(inference_res);
            }
            else if (!m_last_image.empty())
            {
                auto dest = HostCache<float3>();
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
                                               HostCache<float3>(),
                                               inference_res.size() / 3).Wait();
            }
            else
            {
                m_context->CopyBuffer<float3>(0,
                                      dynamic_cast<ClwOutput*>(input_set.at(OutputType::kColor))->data(),
                                      clw_inference_output->data(),
                                      0,
                                      0,
                                      shape.width * shape.height).Wait();
            }
        }
    }
}
