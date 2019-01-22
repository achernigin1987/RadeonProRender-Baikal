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


#include "PostEffects/ML/denoiser_preprocess.h"

#include "CLWBuffer.h"
#include "Output/clwoutput.h"
#include "math/mathutils.h"

#include <RadeonProML.h>

namespace Baikal
{
    namespace PostEffects
    {
        using float3 =  RadeonRays::float3;
        using OutputType = Renderer::OutputType;

        DenoiserPreprocess::DenoiserPreprocess(CLWContext context,
                                               CLProgramManager const *program_manager,
                                               std::uint32_t width,
                                               std::uint32_t height,
                                               std::uint32_t start_spp)
        : DataPreprocess(context, program_manager)
        , m_primitives(CLWParallelPrimitives(context))
        , m_start_spp(start_spp)
        , m_width(width)
        , m_height(height)
        , m_model(Model::kColorAlbedoDepthNormal9)
        , m_context(mlCreateContext())
        {
            switch (m_model)
            {
                case Model::kColorDepthNormalGloss7:
                    m_layout.emplace_back(OutputType::kColor, 3);
                    m_layout.emplace_back(OutputType::kDepth, 1);
                    m_layout.emplace_back(OutputType::kViewShadingNormal, 2);
                    m_layout.emplace_back(OutputType::kGloss, 1);
                    break;
                case Model::kColorAlbedoNormal8:
                    m_layout.emplace_back(OutputType::kColor, 3 );
                    m_layout.emplace_back(OutputType::kAlbedo, 3);
                    m_layout.emplace_back(OutputType::kViewShadingNormal, 2);
                    break;
                case Model::kColorAlbedoDepthNormal9:
                    m_layout.emplace_back(OutputType::kColor, 3 );
                    m_layout.emplace_back(OutputType::kAlbedo, 3);
                    m_layout.emplace_back(OutputType::kDepth, 1);
                    m_layout.emplace_back(OutputType::kViewShadingNormal, 2);
                    break;
            }

            m_cache = CLWBuffer<float>::Create(
                    context,
                    CL_MEM_READ_WRITE,
                    width * height);

            m_channels = 0;
            for (const auto& layer: m_layout)
            {
                m_channels += layer.second;
            }

            m_input = CLWBuffer<float>::Create(
                    context,
                    CL_MEM_READ_WRITE,
                    m_channels * width * height);

            ml_image_info image_info = {ML_FLOAT32, m_width, m_height, m_channels};
            m_image = mlCreateImage(m_context, &image_info);

            if (!m_image)
            {
                throw std::runtime_error("can not create ml_image");
            }
        }


        Image DenoiserPreprocess::MakeInput(PostEffect::InputSet const& inputs)
        {
            auto context = GetContext();
            unsigned channels_count = 0u;
            float real_sample_count = 0.f;
            bool too_few_samples = false;

            for (const auto& desc : m_layout)
            {
                if (too_few_samples)
                {
                    return Image(static_cast<std::uint32_t>(real_sample_count), nullptr);
                }

                auto type = desc.first;
                auto input = inputs.at(type);

                auto clw_output = dynamic_cast<ClwOutput*>(input);
                auto device_mem = clw_output->data();

                unsigned channels_to_copy = 0u;

                switch (type)
                {
                    case OutputType::kColor:
                    {
                        DivideBySampleCount(CLWBuffer<float3>::CreateFromClBuffer(m_cache),
                                            CLWBuffer<float3>::CreateFromClBuffer(device_mem));

                        channels_count += 3;

                        WriteToInputs(m_input,
                                      m_cache,
                                      m_width,
                                      m_height,
                                      channels_count,
                                      m_channels,
                                      0,
                                      4,
                                      3);

                        context.ReadBuffer<float>(0, m_cache, &real_sample_count, 3, 1).Wait();

                        if (real_sample_count < m_start_spp)
                        {
                            too_few_samples = true;
                        }
                        break;
                    }
                    case OutputType::kDepth:
                    {
                        auto normalized_buf = CLWBuffer<cl_float3>::CreateFromClBuffer(m_cache);

                        m_primitives.Normalize(0,
                                               CLWBuffer<cl_float3>::CreateFromClBuffer(device_mem),
                                               normalized_buf,
                                               (int)device_mem.GetElementCount() / sizeof(cl_float3));

                        WriteToInputs(m_input,
                                      CLWBuffer<float>::CreateFromClBuffer(normalized_buf),
                                      m_width,
                                      m_height,
                                      channels_count,
                                      m_channels,
                                      0,
                                      4,
                                      1).Wait();

                        channels_count += 1;
                        break;
                    }
                    case OutputType::kViewShadingNormal:
                    {
                        channels_to_copy = 2;
                        break;
                    }
                    case OutputType::kGloss:
                    {
                        channels_to_copy = 1;
                        break;
                    }
                    case OutputType::kAlbedo:
                    {
                        channels_to_copy = 3;
                        break;
                    }
                    default:
                        break;
                }

                if (channels_to_copy)
                {
                    DivideBySampleCount(CLWBuffer<float3>::CreateFromClBuffer(m_cache),
                                        CLWBuffer<float3>::CreateFromClBuffer(device_mem));

                    WriteToInputs(m_input,
                                  m_cache,
                                  m_width,
                                  m_height,
                                  channels_count,
                                  m_channels,
                                  0,
                                  4,
                                  channels_to_copy).Wait();

                    channels_count += channels_to_copy;
                }
            }

            size_t image_size = 0;
            auto host_buffer = mlMapImage(m_image, &image_size);

            if (!host_buffer || image_size == 0)
            {
                throw std::runtime_error("map operation failed");
            }

            context.ReadBuffer<float>(0,
                    m_input,
                    static_cast<float*>(host_buffer),
                    m_input.GetElementCount()).Wait();

            if (mlUnmapImage(m_image, host_buffer) != ML_OK)
            {
                throw std::runtime_error("unmap operation failed");
            }

            return Image(static_cast<std::uint32_t>(real_sample_count), m_image);
        }

        std::set<Renderer::OutputType> DenoiserPreprocess::GetInputTypes() const
        {
            switch (m_model)
            {
                case Model::kColorDepthNormalGloss7:
                    return std::set<Renderer::OutputType>(
                            {
                                    Renderer::OutputType::kColor,
                                    Renderer::OutputType::kDepth,
                                    Renderer::OutputType::kViewShadingNormal,
                                    Renderer::OutputType::kGloss,
                            });

                case Model::kColorAlbedoNormal8:
                    return std::set<Renderer::OutputType>(
                            {
                                    Renderer::OutputType::kColor,
                                    Renderer::OutputType::kAlbedo,
                                    Renderer::OutputType::kViewShadingNormal,
                            });
                case Model::kColorAlbedoDepthNormal9:
                    return std::set<Renderer::OutputType>(
                            {
                                    Renderer::OutputType::kColor,
                                    Renderer::OutputType::kAlbedo,
                                    Renderer::OutputType::kDepth,
                                    Renderer::OutputType::kViewShadingNormal,
                            });
                default:
                    throw std::runtime_error("Model is not supported");
            }
        }

        void DenoiserPreprocess::DivideBySampleCount(CLWBuffer<float3> dst,
                                                     CLWBuffer<float3> src)
        {
            assert (dst.GetElementCount() >= src.GetElementCount());

            auto division_kernel = GetKernel("DivideBySampleCount");

            // Set kernel parameters
            int argc = 0;
            division_kernel.SetArg(argc++, dst);
            division_kernel.SetArg(argc++, src);
            division_kernel.SetArg(argc++, (int)src.GetElementCount());

            // run DivideBySampleCount kernel
            auto thread_num = ((src.GetElementCount() + 63) / 64) * 64;
            GetContext().Launch1D(0,
                                  thread_num,
                                  64,
                                  division_kernel);
        }
    }
}
