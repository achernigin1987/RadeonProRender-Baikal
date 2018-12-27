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
#include "inference.h"

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

                ml_image_info image_info = {ML_FLOAT32, width, height, 3};
                ml_image_info output_info = {ML_FLOAT32, 2 * width, 2 * height, 3};

                return std::make_unique<Inference>(model_path,
                                                   image_info,
                                                   output_info,
                                                   gpu_memory_fraction,
                                                   visible_devices);
            }
        }

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
                m_width = color_aov->width();
                m_height = color_aov->height();
                m_inference = CreateInference(gpu_memory_fraction,
                                              visible_devices,
                                              m_width,
                                              m_height);

                m_device_cache.reset();
                m_device_cache = std::make_unique<CLWBuffer<float3>>(
                    CLWBuffer<float3>::Create(*m_context, CL_MEM_READ_WRITE,
                                              m_width * m_height));

                m_resizer_cache.reset();
                m_resizer_cache = std::make_unique<CLWBuffer<float3>>(
                        CLWBuffer<float3>::Create(*m_context, CL_MEM_READ_WRITE,
                                                  2 * m_width * m_height)
                        );

                m_input_cache.reset();
                m_input_cache = std::make_unique<CLWBuffer<float>>(
                    CLWBuffer<float>::Create(*m_context,
                                             CL_MEM_READ_WRITE,
                                             3 * m_width * m_height));

                m_cache.resize(4 * output.width() * output.height());

                m_last_denoised_image.reset();

                m_last_denoised_image = std::make_unique<CLWBuffer<float3>>(
                        CLWBuffer<float3>::Create(
                                *m_context,
                                CL_MEM_READ_WRITE,
                                4 * m_width * m_height));

                m_has_denoised_image = false;
            }

            auto clw_input = dynamic_cast<ClwOutput*>(color_aov);

            if (clw_input== nullptr)
            {
                throw std::runtime_error("SuperRes::Apply(..): incorrect input");
            }

            Tonemap(*m_device_cache, clw_input->data());

            auto copy_kernel = GetKernel("CopyInterleaved");

            int argc = 0;
            copy_kernel.SetArg(argc++, *m_input_cache); // dst
            copy_kernel.SetArg(argc++, *m_device_cache); // src
            copy_kernel.SetArg(argc++, m_width); // dst_width
            copy_kernel.SetArg(argc++, m_height); // dst_height
            copy_kernel.SetArg(argc++, 0); // dst_channels_offset
            copy_kernel.SetArg(argc++, 3); // dst_channels_num
            // input and output buffers have the same width in pixels
            copy_kernel.SetArg(argc++, m_width); // src_width
            // input and output buffers have the same height in pixels
            copy_kernel.SetArg(argc++, m_height); // src_height
            copy_kernel.SetArg(argc++, 0); // src_channels_offset
            copy_kernel.SetArg(argc++, 4); // src_channels_num
            copy_kernel.SetArg(argc++, 3); // channels_to_copy
            copy_kernel.SetArg(argc++, nullptr); // out_sample_count

            // run copy_kernel
            auto thread_num = ((m_width * m_height + 63) / 64) * 64;
            m_context->Launch1D(0,
                                thread_num,
                                64,
                                copy_kernel);

            RadeonRays::float3 real_sample_count = .0f;
            m_context->ReadBuffer<float3>(0, clw_input->data(), &real_sample_count, 1).Wait();
            auto sample_count = static_cast<unsigned>(real_sample_count.w);

            // reset denoised image if
            if (sample_count == 1)
            {
                m_start_seq_num = m_last_seq_num + 1;
                m_has_denoised_image = false;
            }

            auto input = m_inference->GetInputData();

            input.tag = ++m_last_seq_num;

            size_t input_size;
            auto input_data = static_cast<float*>(mlMapImage(input.image, &input_size));

            if (input_data == nullptr)
            {
                throw std::runtime_error("ml buffer map operation failed");
            }

            m_context->ReadBuffer<float>(0,
                                         *m_input_cache,
                                         input_data,
                                         input_size / sizeof(float)).Wait();

            mlUnmapImage(input.image, input_data);

            // push tensor in model queue
            m_inference->PushInput(std::move(input));

            auto clw_output = dynamic_cast<ClwOutput*>(&output);
            if (clw_output == nullptr)
            {
                throw std::runtime_error("SuperRes::Apply(..): incorrect output");
            }

            auto output_device_mem = clw_output->data();

            // get another tensor from model queue
            auto model_output = m_inference->PopOutput();

            if (model_output.image != ML_INVALID_HANDLE && model_output.tag >= m_start_seq_num)
            {
                size_t output_size;
                auto output_data = static_cast<float*>(
                        mlMapImage(model_output.image, &output_size));

                for (auto i = 0u; i < output_device_mem.GetElementCount(); i++)
                {
                    // 4th component (w) is not written here because
                    // it is saved from the previous reading
                    m_cache[4 * i] = std::max(output_data[3 * i], 0.f);
                    m_cache[4 * i + 1] = std::max(output_data[3 * i + 1], 0.f);
                    m_cache[4 * i + 2] = std::max(output_data[3 * i + 2], 0.f);
                    m_cache[4 * i + 3] = 1;
                }

                mlUnmapImage(model_output.image, output_data);

                // if returned tensor is empty return black image
                m_context->WriteBuffer<float3>(0,
                                               *m_last_denoised_image,
                                               reinterpret_cast<float3 *>(m_cache.data()),
                                               output_device_mem.GetElementCount()).Wait();

                m_context->CopyBuffer<float3>(0,
                                              *m_last_denoised_image,
                                              output_device_mem,
                                              0 /* srcOffset */,
                                              0 /* destOffset */,
                                              m_last_denoised_image->GetElementCount()).Wait();

                m_has_denoised_image = true;
            }
            else if (m_has_denoised_image)
            {
                m_context->CopyBuffer<float3>(0,
                                              *m_last_denoised_image,
                                              output_device_mem,
                                              0 /* srcOffset */,
                                              0 /* destOffset */,
                                              m_last_denoised_image->GetElementCount()).Wait();
            }
            else
            {
                auto scale_x = GetKernel("BicubicUpScaleX_x2");

                int argc = 0;
                scale_x.SetArg(argc++, *m_resizer_cache);
                scale_x.SetArg(argc++, *m_device_cache);
                scale_x.SetArg(argc++, m_width);
                scale_x.SetArg(argc++, m_height);

                // run BicubicUpScaleX_x2 kernel
                auto thread_num = ((2 * m_width * m_height + 63) / 64) * 64;
                m_context->Launch1D(0,
                                    thread_num,
                                    64,
                                    scale_x);

                auto scale_y = GetKernel("BicubicUpScaleY_x2");

                argc = 0;
                scale_y.SetArg(argc++, output_device_mem);
                scale_y.SetArg(argc++, *m_resizer_cache);
                scale_y.SetArg(argc++, 2 * m_width);
                scale_y.SetArg(argc++, m_height);

                // run BicubicUpScaleY_x2 kernel
                thread_num = ((4 * m_width * m_height + 63) / 64) * 64;
                m_context->Launch1D(0,
                                    thread_num,
                                    64,
                                    scale_y).Wait();
            }
        }

        PostEffect::InputTypes SuperRes::GetInputTypes() const
        {
            return std::set<Renderer::OutputType>({Renderer::OutputType::kColor});
        }

        void SuperRes::Tonemap(CLWBuffer<RadeonRays::float3> dst, CLWBuffer<RadeonRays::float3> src)
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
            m_context->Launch1D(0,
                                thread_num,
                                64,
                                tonemapping);
        }
    }
}