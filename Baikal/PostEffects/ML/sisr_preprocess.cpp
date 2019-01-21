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
#include "inference.h"

#include "Output/clwoutput.h"
#include <CLW.h>

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif


namespace Baikal
{
    namespace PostEffects
    {
        using uint32_t = std::uint32_t;
        using float3 =  RadeonRays::float3;

        SisrPreprocess::SisrPreprocess(CLWContext context,
                                               Baikal::CLProgramManager const *program_manager,
                                               std::uint32_t width,
                                               std::uint32_t height,
                                               std::uint32_t start_spp)
#ifdef BAIKAL_EMBED_KERNELS
        : ClwClass(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers)
#else
        : ClwClass(context, program_manager, "../Baikal/Kernels/CL/denoise.cl")
#endif
        , m_width(width),
        , m_height(height)
        , m_spp(start_spp)
        , m_context(mlCreateContext())
        {
            auto context = GetContext();

            m_cache = CLWBuffer::Create<float3>(context,
                                                CL_MEM_READ_WRITE,
                                                width * height);

            m_input = CLWBuffer::Create<float>(context,
                                                   CL_MEM_READ_WRITE,
                                                   3 * width * height);

            ml_image_info image_info = {ML_FLOAT32, width, height, 3};
            m_image = mlCreateImage(m_context, &image_info);

            if (!m_image)
            {
                throw std::runtime_error("can not create ml_image");
            }
        }

        ml_image SisrPreprocess::MakeInput(PostEffect::InputSet const& inputs) {
            auto color_aov = input_set.begin()->second;

            auto clw_input = dynamic_cast<ClwOutput *>(color_aov);

            if (clw_input == nullptr) {
                throw std::runtime_error("SisrPreprocess::MakeInput(..): incorrect input");
            }

            // read spp from first pixel as 4th channel
            RadeonRays::float3 pixel = .0f;
            m_context->ReadBuffer<float3>(0, clw_input->data(), &pixel, 1).Wait();
            auto sample_count = static_cast<unsigned>(real_sample_count.w);

            if (m_start_spp > sample_count)
            {
                return nullptr;
            }

            Tonemap(m_cache, clw_input->data());

            // delete 4th channel
            WriteToInputs(m_input,
                          m_cache,
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

            return m_image;
        }
//
//        void SisrPreprocess::Resize_x2(CLWBuffer<RadeonRays::float3> dst, CLWBuffer<RadeonRays::float3> src)
//        {
//            auto context = GetContext();
//
//            if (m_resizer_cache == nullptr ||
//                m_resizer_cache->GetElementCount() < 2 * src.GetElementCount())
//            {
//                m_resizer_cache.reset();
//                m_resizer_cache = std::make_unique<CLWBuffer<float3>>(
//                        CLWBuffer<float3>::Create(context, CL_MEM_READ_WRITE,
//                                                  2 * src.GetElementCount())
//                );
//            }
//
//            auto scale_x = GetKernel("BicubicUpScaleX_x2");
//
//            int argc = 0;
//            scale_x.SetArg(argc++, *m_resizer_cache);
//            scale_x.SetArg(argc++, src);
//            scale_x.SetArg(argc++, m_width);
//            scale_x.SetArg(argc++, m_height);
//
//            // run BicubicUpScaleX_x2 kernel
//            auto thread_num = ((2 * m_width * m_height + 63) / 64) * 64;
//            context.Launch1D(0,
//                             thread_num,
//                             64,
//                             scale_x);
//
//            auto scale_y = GetKernel("BicubicUpScaleY_x2");
//
//            argc = 0;
//            scale_y.SetArg(argc++, dst);
//            scale_y.SetArg(argc++, *m_resizer_cache);
//            scale_y.SetArg(argc++, 2 * m_width);
//            scale_y.SetArg(argc++, m_height);
//
//            // run BicubicUpScaleY_x2 kernel
//            thread_num = ((4 * m_width * m_height + 63) / 64) * 64;
//            context.Launch1D(0,
//                             thread_num,
//                             64,
//                             scale_y).Wait();
//        }
    }
}