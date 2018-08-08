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

#include "CLWBuffer.h"
#include "math/mathutils.h"

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
            std::unique_ptr<Inference> CreateDenoiserInference(
                    MLDenoiserInputs inputs,
                    float gpu_memory_fraction,
                    std::string const &visible_devices,
                    std::size_t width,
                    std::size_t height) {
                std::string model_path;
                std::size_t input_channels;
                switch (inputs) {
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

        MLDenoiser::MLDenoiser(const CLWContext& context)
                   : m_inputs(MLDenoiserInputs::kColorAlbedoNormal8)
        {
            RegisterParameter("gpu_memory_fraction", .1f);
            RegisterParameter("visible_devices", std::string());

            m_context = std::make_unique<CLWContext>(context);
            m_primitives = std::make_unique<CLWParallelPrimitives>(context);

            // compute memory layout
            switch (m_inputs)
            {
            case MLDenoiserInputs::kColorDepthNormalGloss7:
                m_layout.emplace_back(OutputType::kColor, 3);
                m_layout.emplace_back(OutputType::kDepth, 1);
                m_layout.emplace_back(OutputType::kViewShadingNormal, 2);
                m_layout.emplace_back(OutputType::kGloss, 1);
                break;

            case MLDenoiserInputs::kColorAlbedoNormal8:
                m_layout.emplace_back(OutputType::kColor, 3 );
                m_layout.emplace_back(OutputType::kAlbedo, 3);
                m_layout.emplace_back(OutputType::kViewShadingNormal, 2);
                break;
            }
        }

        void MLDenoiser::InitDenoiserInference()
        {
            auto gpu_memory_fraction = GetParameter("gpu_memory_fraction").GetFloat();
            auto visible_devices = GetParameter("visible_devices").GetString();

            m_inference = CreateDenoiserInference(m_inputs,
                                                  gpu_memory_fraction,
                                                  visible_devices,
                                                  m_width, m_height);

            // Realloc cache if needed
            auto shape = m_inference->GetInputShape();

            size_t bytes_count = sizeof(Tensor::ValueType) * shape.width * shape.height * shape.channels;

            if (m_host_cache.size() < bytes_count)
            {
                m_device_cache = std::make_unique<CLWBuffer<char>>(
                        CLWBuffer<char>::Create(*m_context, CL_MEM_READ_WRITE, bytes_count));
                m_device_depth_cache = std::make_unique<CLWBuffer<RadeonRays::float3>>(
                        CLWBuffer<RadeonRays::float3>::Create(*m_context, CL_MEM_READ_WRITE,
                                                              shape.width * shape.height));

                m_host_cache.resize(bytes_count);
            }
        }

        PostEffect::InputTypes MLDenoiser::GetInputTypes() const
        {
            switch (m_inputs) {
                case MLDenoiserInputs::kColorDepthNormalGloss7:
                    return std::set<Renderer::OutputType>(
                            {
                                    Renderer::OutputType::kColor,
                                    Renderer::OutputType::kDepth,
                                    Renderer::OutputType::kViewShadingNormal,
                                    Renderer::OutputType::kGloss,
                            });

                case MLDenoiserInputs::kColorAlbedoNormal8:
                    return std::set<Renderer::OutputType>(
                            {
                                    Renderer::OutputType::kColor,
                                    Renderer::OutputType::kAlbedo,
                                    Renderer::OutputType::kViewShadingNormal,
                            });
                default:
                    throw std::runtime_error("Model is not supported");
            }
        }

        void MLDenoiser::Apply(InputSet const& input_set, Output& output)
        {
            if (m_width != input_set.begin()->second->width() ||
                m_height != input_set.begin()->second->height())
            {
                m_width = input_set.begin()->second->width();
                m_height = input_set.begin()->second->height();
                m_is_dirty = true;
            }

            if (m_is_dirty)
            {
                InitDenoiserInference();
                m_is_dirty = false;
            }

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
                        dest[0] = std::pow(source->x / source->w, 1.f / 2.2f);
                        dest[1] = std::pow(source->y / source->w, 1.f / 2.2f);
                        dest[2] = std::pow(source->z / source->w, 1.f / 2.2f);
                        dest += shape.channels;
                        ++source;
                    }
                    break;
                }
                case OutputType::kDepth:
                {
                    m_context->ReadBuffer<float3>(0,
                                                  device_mem,
                                                  HostCache<float3>(),
                                                  device_mem.GetElementCount()).Wait();

                    m_context->WriteBuffer<RadeonRays::float3>(0,
                                                               *m_device_depth_cache,
                                                               reinterpret_cast<RadeonRays::float3*>(m_host_cache.data()),
                                                               device_mem.GetElementCount()).Wait();

                    auto normalized_buf = CLWBuffer<cl_float3>::CreateFromClBuffer(*m_device_cache);

                    m_primitives->Normalize(0,
                                            CLWBuffer<cl_float3>::CreateFromClBuffer(*m_device_depth_cache),
                                            normalized_buf,
                                            (int)device_mem.GetElementCount()).Wait();

                    m_context->ReadBuffer<RadeonRays::float3>(0,
                                                CLWBuffer<RadeonRays::float3>::CreateFromClBuffer(normalized_buf),
                                                (RadeonRays::float3*)m_host_cache.data(),
                                                device_mem.GetElementCount()).Wait();
                    auto t = device_mem.GetElementCount();
                    auto dest = host_mem;
                    auto source = HostCache<float3>();
                    for (auto i = 0u; i < t; ++i)
                    {
                        *dest = source->x;
                        dest += shape.channels;
                        ++source;
                    }
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
                            dest[0] = RadeonRays::clamp(source->x / source->w, 0.f, 1.f);
                            dest[1] = RadeonRays::clamp(source->y / source->w, 0.f, 1.f);
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
                m_last_denoised_image = {};
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
                    auto constexpr gamma = 2.2f;
                    dest->x = std::pow(*source++, gamma);
                    dest->y = std::pow(*source++, gamma);
                    dest->z = std::pow(*source++, gamma);
                    dest->w = 1;
                    ++dest;
                }

                m_context->WriteBuffer<float3>(0,
                    clw_inference_output->data(),
                    HostCache<float3>(),
                    inference_res.size() / 3).Wait();

                m_last_denoised_image = std::move(inference_res);
            }
            else if (!m_last_denoised_image.empty())
            {
                auto dest = HostCache<float3>();
                auto source = m_last_denoised_image.data();
                for (auto i = 0u; i < shape.width * shape.height; ++i)
                {
                    auto constexpr gamma = 2.2f;
                    dest->x = std::pow(*source++, gamma);
                    dest->y = std::pow(*source++, gamma);
                    dest->z = std::pow(*source++, gamma);
                    dest->w = 1;
                    ++dest;
                }

                m_context->WriteBuffer<float3>(0,
                                               clw_inference_output->data(),
                                               HostCache<float3>(),
                                               m_last_denoised_image.size() / 3).Wait();
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

        void MLDenoiser::Update(Camera* camera, unsigned int samples)
        {

        }

        void MLDenoiser::SetParameter(std::string const& name, Param value)
        {
            auto param = GetParameter(name);
            PostEffect::SetParameter(name, value);
            m_is_dirty = true;
        }
    }
}
