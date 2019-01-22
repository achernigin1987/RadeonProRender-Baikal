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

#include "sisr_preprocess.h"
#include "Output/clwoutput.h"
#include "CLWBuffer.h"

namespace Baikal
{
    namespace PostEffects
    {
        using uint32_t = std::uint32_t;
        using float3 =  RadeonRays::float3;

        SisrPreprocess::SisrPreprocess(CLWContext context,
                                       Baikal::CLProgramManager const *program_manager,
                                       std::uint32_t start_spp)
        : DataPreprocess(context, program_manager, start_spp)
        , m_context(mlCreateContext())
        {}

        void SisrPreprocess::Init(std::uint32_t width, std::uint32_t height)
        {
            m_cache = CLWBuffer<float3>::Create(GetContext(),
                                                CL_MEM_READ_WRITE,
                                                width * height);

            m_input = CLWBuffer<float>::Create(GetContext(),
                                               CL_MEM_READ_WRITE,
                                               3 * width * height);

            ml_image_info image_info = {ML_FLOAT32, width, height, 3};
            m_image = mlCreateImage(m_context, &image_info);

            if (!m_image)
            {
                throw std::runtime_error("can not create ml_image");
            }
        }

        Image SisrPreprocess::MakeInput(PostEffect::InputSet const& inputs)
        {
            auto color_aov = inputs.begin()->second;

            if (!m_is_init)
            {
                m_width = color_aov->width();
                m_height = color_aov->height();
                Init(m_width, m_height);
                m_is_init = true;
            }

            auto clw_input = dynamic_cast<ClwOutput *>(color_aov);

            if (clw_input == nullptr) {
                throw std::runtime_error("SisrPreprocess::MakeInput(..): incorrect input");
            }

            auto context = GetContext();

            // read spp from first pixel as 4th channel
            RadeonRays::float3 pixel = .0f;
            context.ReadBuffer<float3>(0, clw_input->data(), &pixel, 1).Wait();
            auto sample_count = static_cast<unsigned>(pixel.w);

            if (m_start_spp > sample_count)
            {
                return Image(sample_count, nullptr);
            }

            Tonemap(m_cache, clw_input->data());

            // delete 4th channel
            WriteToInputs(m_input,
                          CLWBuffer<float>::CreateFromClBuffer(m_cache),
                          m_width,
                          m_height,
                          0,  // dst channels offset
                          3,  // dst channels num
                          0,  // src channels offset
                          4,  // src channels num
                          3); // channels to copy

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

            return Image(sample_count, m_image);
        }

        void SisrPreprocess::Tonemap(CLWBuffer<RadeonRays::float3> dst, CLWBuffer<RadeonRays::float3> src)
        {
            assert (dst.GetElementCount() >= src.GetElementCount());

            auto tonemapping = GetKernel("TonemapExponential");

            // Set kernel parameters
            int argc = 0;
            tonemapping.SetArg(argc++, dst);
            tonemapping.SetArg(argc++, src);
            tonemapping.SetArg(argc++, (int)src.GetElementCount());

            // run DivideBySampleCount kernel
            auto thread_num = ((src.GetElementCount() + 63) / 64) * 64;
            GetContext().Launch1D(0,
                                  thread_num,
                                  64,
                                  tonemapping);
        }

        std::set<Renderer::OutputType> SisrPreprocess::GetInputTypes() const
        {
            return std::set<Renderer::OutputType>({Renderer::OutputType::kColor});
        }
    }
}