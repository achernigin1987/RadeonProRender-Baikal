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
#include <ml.h>

namespace Baikal
{
    namespace PostEffects
    {
        Inference::Inference(std::string const& model_path,
                             // input shapes
                             Tensor::Shape const& input_shape,
                             Tensor::Shape const& output_shape,
                             // model params
                             float gpu_memory_fraction,
                             std::string const& visible_devices,
                             std::string const& input_node,
                             std::string const& output_node)
                : m_go_flag(true),
                  m_model(model_path,
                          input_node,
                          output_node,
                          gpu_memory_fraction,
                          visible_devices)
        {
            // specify input tensor shape for model
            if (mlGetModelInfo(m_model.GetModel(), &m_input_desc, NULL) != ML_OK)
            {
                throw std::runtime_error("can not get input shape");
            }

            if (m_input_desc.channels != input_shape.channels)
            {
                throw std::runtime_error(
                        "passed input channels number doesn't correspond"
                        " to model input channels number");
            }

            m_input_desc.width = input_shape.width;
            m_input_desc.height = input_shape.height;

            if (mlSetModelInputInfo(m_model.GetModel(), &m_input_desc) != ML_OK)
            {
                throw std::runtime_error(
                        "can not set input shape to model due to unknown reason");
            }

            // get output tensor shape for out model
            if (mlGetModelInfo(m_model.GetModel(), NULL, &m_output_desc) != ML_OK)
            {
                throw std::runtime_error("can not get input shape");
            }

            m_worker = std::thread(&Inference::DoInference, this);
        }

        Inference::~Inference()
        {
            Shutdown();
        }

        Tensor::Shape Inference::GetInputShape() const
        {
            return { m_input_desc.width, m_input_desc.height, m_input_desc.channels };
        }

        Tensor::Shape Inference::GetOutputShape() const
        {
            return { m_output_desc.width, m_output_desc.height, m_output_desc.channels };
        }

        Data Inference::GetInputData()
        {
            return AllocData(m_input_desc);
        }

        Data Inference::AllocData(ml_image_info const &info)
        {
            auto input_image = m_model.CreateImage(info);

            if (input_image == ML_INVALID_HANDLE)
            {
                throw std::runtime_error("can not create input image");
            }

            size_t size;
            auto input_data = mlMapImage(input_image, &size);

            if (input_data == NULL)
            {
                throw std::runtime_error("can not map image");
            }

            return {input_data, input_image};
        }

        void Inference::PushInput(Data&& tensor)
        {
            m_input_queue.push(std::move(tensor));
        }

        Data Inference::PopOutput()
        {
            Data output_tensor;
            m_output_queue.try_pop(output_tensor);
            return output_tensor;
        }


        void Inference::DoInference()
        {
            for (;;)
            {
                Data input;
                m_input_queue.wait_and_pop(input);

                if (input.is_empty)
                {
                    break;
                }

                if (m_input_queue.size() > 0)
                {
                    continue;
                }

                Data output = AllocData(m_output_desc);

                mlUnmapImage(output.gpu_data, output.cpu_data);

                if (mlInfer(m_model.GetModel(), input.gpu_data, output.gpu_data) != ML_OK)
                {
                    std::cerr << "Can't perform inference"
                    continue;
                }

                m_output_queue.push(std::move(output));
            }
        }

        void Inference::Shutdown()
        {
            m_input_queue.push({true, nullptr, nullptr});
            m_worker.join();
        }
    }
}