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

#include "ml_post_effect.h"


#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#endif

namespace Baikal
{
    namespace PostEffects {

        MlPostEffect::MlPostEffect(const CLWContext &context, const CLProgramManager *program_manager)
#ifdef BAIKAL_EMBED_KERNELS
        : ClwPostEffect(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers),
#else
        : ClwPostEffect(context, program_manager, "../Baikal/Kernels/CL/denoise.cl"),
#endif
          m_is_dirty(true)
        {
            m_inference = nullptr;
            m_context = std::make_unique<CLWContext>(context);

            RegisterParameter("gpu_memory_fraction", .7f);
            RegisterParameter("visible_devices", std::string());
        }


        void MlPostEffect::Init(InputSet const& input_set, Output& output)
        {
            auto aov = input_set.begin()->second;

            auto gpu_memory_fraction = GetParameter("gpu_memory_fraction").GetFloat();
            auto visible_devices = GetParameter("visible_devices").GetString();

            m_width = color_aov->width();
            m_height = color_aov->height();

            m_inference = CreateInference(gpu_memory_fraction,
                                          visible_devices,
                                          m_width,
                                          m_height);

            auto shape = m_inference->GetInputShape();

            m_device_buf = std::make_unique<CLWBuffer<float>>(
                    CLWBuffer<float>::Create(*m_context,
                                             CL_MEM_READ_WRITE,
                                             shape.channels * shape.width * shape.height));
        }


        void MlPostEffect::Apply(InputSet const& input_set, Output& output)
        {
            auto start_spp = GetParameter("start_spp").GetUint();

            if (m_width != input_set.begin()->second->width() ||
                m_height != input_set.begin()->second->height())
            {
                m_is_dirty = true;
            }

            if (m_is_dirty)
            {
                Init(input_set, output);
                m_is_dirty = false;
            }

            auto shape = m_inference->GetInputShape();


            unsigned sample_count = 0;
            unsigned channels_count = 0u;
            bool too_few_samples = false;

            // Get input buffer using custom user specified function
            auto device_buffer = GetInput(input_set);

            if (PrepeareInput(m_device_buf, input_set))
            {
                m_start_seq_num = m_last_seq_num + 1;
            }
            else
            {
                size_t input_size;
                auto input = m_inference->GetInputData();
                auto input_data = static_cast<float*>(mlMapImage(input.image, &input_size));

                if (input_data == nullptr)
                {
                    throw std::runtime_error("map input image has failed");
                }

                m_context->ReadBuffer<float>(0,
                                             *m_device_buf,
                                             input_data,
                                             m_device_tensor->GetElementCount()).Wait();

                input.tag = ++m_last_seq_num;
                mlUnmapImage(input.image, input_data);

                m_inference->PushInput(std::move(input));
            }

            // process output
            auto clw_inference_output = dynamic_cast<ClwOutput*>(&output);

            if (!clw_inference_output)
            {
                throw std::runtime_error("MLDenoiser::Apply(...): can not cast output");
            }

            SetOutput(m_inference->PopOutput(), output);
        }


        void MLDenoiser::SetParameter(std::string const& name, Param value)
        {
            auto param = GetParameter(name);
            PostEffect::SetParameter(name, value);
            m_is_dirty = true;
        }
    }
}