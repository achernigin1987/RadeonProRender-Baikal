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


#include "inference.h"
#include <cassert>

namespace Baikal
{
    namespace PostEffects
    {
        Inference::Inference(std::string const& model_path,
                             std::string const& input_node,
                             std::string const& output_node,
                             float gpu_memory_fraction,
                             std::string const& visible_devices,
                             std::size_t in_width,
                             std::size_t in_height,
                             std::size_t out_width,
                             std::size_t out_height,
                             std::size_t input_channels)
                : m_model(model_path, input_node, output_node, gpu_memory_fraction, visible_devices)
                , m_in_width(in_width)
                , m_in_height(in_height)
                , m_out_width(out_width)
                , m_out_height(out_height)
                , m_input_channels(input_channels)
                , m_worker(&Inference::DoInference, this)
        { }

        Inference::~Inference()
        {
            Shutdown();
        }

        Tensor::Shape Inference::GetInputShape() const
        {
            return { m_in_width, m_in_height, m_input_channels };
        }

        Tensor::Shape Inference::GetOutputShape() const
        {
            return { m_out_width, m_out_height, m_output_channels };
        }

        Tensor Inference::GetInputTensor()
        {
            return AllocTensor(m_in_width, m_in_height, m_input_channels);
        }

        void Inference::PushInput(Tensor&& tensor)
        {
            assert(tensor.shape() == GetInputShape());
            m_input_queue.push(std::move(tensor));
        }

        Tensor Inference::PopOutput()
        {
            Tensor output_tensor;
            m_output_queue.try_pop(output_tensor);
            return output_tensor;
        }

        Tensor Inference::AllocTensor(std::size_t width, std::size_t height, std::size_t channels)
        {
            auto deleter = [](Tensor::ValueType* data)
            {
                delete[] data;
            };
            size_t size = width * height * channels;
            return Tensor(Tensor::Data(new Tensor::ValueType[size], deleter),
                          { width, height, channels });
        }

        void Inference::Shutdown()
        {
            m_input_queue.push(Tensor());
            m_worker.join();
        }
    }
}