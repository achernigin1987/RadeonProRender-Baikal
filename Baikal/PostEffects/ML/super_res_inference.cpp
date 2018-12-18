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

#include "super_res_inference.h"
#include <iostream>
#include <cassert>
#include <cstring>

namespace Baikal
{
    namespace PostEffects
    {
        using uint32_t = std::uint32_t;

        SuperResInference::SuperResInference(std::string const& model_path,
                                             float gpu_memory_fraction,
                                             std::string const& visible_devices,
                                             std::size_t width,
                                             std::size_t height)
        : Inference(model_path,
                    "Generator/inputs",
                    "Generator/outputs",
                    gpu_memory_fraction,
                    visible_devices,
                    width,
                    height,
                    2 * width,
                    2 * height,
                    3)
        { }


        void SuperResInference::DoInference()
        {
            for (;;)
            {
                Tensor input_tensor;
                m_input_queue.wait_and_pop(input_tensor);
                if (input_tensor.empty())
                {
                    break;
                }

                if (m_input_queue.size() > 0)
                {
                    continue;
                }

                Tensor output_tensor = AllocTensor(m_out_width, m_out_height, 3);

                m_model->infer(
                        input_tensor.data(),
                        m_in_width,
                        m_in_height,
                        m_input_channels,
                        output_tensor.data());

                output_tensor.tag = input_tensor.tag;
                m_output_queue.push(std::move(output_tensor));
            }
        }
    }
}