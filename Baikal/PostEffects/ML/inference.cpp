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

#include "PostEffects/ML/error_handler.h"
#include "inference.h"
#include "RadeonProML.h"

#include <cassert>

namespace Baikal
{
    namespace PostEffects
    {
        Inference::Inference(ModelHolder* model_holder, 
                             size_t input_height,
                             size_t input_width,
                             bool every_frame)
            : m_model_holder(model_holder)
            , m_every_frame(every_frame)
        {
            CheckStatus(mlGetModelInfo(m_model_holder->GetModel(), &m_input_info, nullptr));
            // Set unspecified input tensor dimensions
            m_input_info.width = input_width;
            m_input_info.height = input_height;
            CheckStatus(mlSetModelInputInfo(m_model_holder->GetModel(), &m_input_info));

            // Get output tensor shape for model
            CheckStatus(mlGetModelInfo(m_model_holder->GetModel(), nullptr, &m_output_info));

            if (m_every_frame)
            {
                m_output_image = m_model_holder->CreateImage(m_output_info, ML_READ_WRITE);
            }
            else
            {
                m_worker = std::thread(&Inference::DoAsyncInference, this);
            }
        }

        Inference::~Inference()
        {
            Shutdown();
        }

        ml_image_info Inference::GetInputInfo() const
        {
            return m_input_info;
        }

        ml_image_info Inference::GetOutputInfo() const
        {
            return m_output_info;
        }

        Image Inference::GetInputData()
        {
            return {0, m_model_holder->CreateImage(m_input_info, ML_READ_WRITE) };
        }

        void Inference::PushInput(Image&& image)
        {
            m_input_queue.push(std::move(image));
        }

        Image Inference::TryPopOutput()
        {
            Image output_tensor = {0, nullptr};
            m_output_queue.try_pop(output_tensor);
            return output_tensor;
        }

        Image Inference::PopOutput()
        {
            Image output_tensor = { 0, nullptr };
            m_output_queue.wait_and_pop(output_tensor);
            return output_tensor;
        }

        Image Inference::Infer(const Image& input)
        {
            CheckStatus(mlInfer(m_model_holder->GetModel(), input.image, m_output_image));
            return Image(static_cast<std::uint32_t>(input.tag), m_output_image);
        }

        void Inference::DoAsyncInference()
        {
            for (;;)
            {
                Image input;
                m_input_queue.wait_and_pop(input);

                if (m_input_queue.size() > 0)
                {
                    continue;
                }

                if (input.image == nullptr)
                {
                    break;
                }

                Image output = { input.tag, m_model_holder->CreateImage(m_output_info, ML_READ_WRITE) };

                CheckStatus(mlInfer(m_model_holder->GetModel(), input.image, output.image));

                m_output_queue.push(std::move(output));
            }
        }

        void Inference::Shutdown()
        {
            if (m_every_frame)
            {
                mlReleaseImage(m_output_image);
            }
            else
            {
                m_input_queue.push({ 0, nullptr });
                m_worker.join();
            }
        }
    }
}
