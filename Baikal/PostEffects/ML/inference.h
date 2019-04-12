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


#pragma once

#include "PostEffects/ML/model_holder.h"

#include "../RadeonRays/RadeonRays/src/async/thread_pool.h"
#include "image.h"

#include <memory>
#include <string>
#include <thread>


namespace Baikal
{
    namespace PostEffects
    {
        class Inference
        {
        public:
            using Ptr = std::unique_ptr<Inference>;

            Inference(ModelHolder* model_holder,
                      size_t input_height,
                      size_t input_width,
                      bool every_frame);

            virtual ~Inference();

            ml_image_info GetInputInfo() const;
            ml_image_info GetOutputInfo() const;

            Image GetInputData();

            void PushInput(Image&& image);
            //
            // Try to pop output image. 
            // Returns empty image, if there are no infered element in the queue
            Image TryPopOutput(); 
            //
            // Wait and pop output image. 
            Image PopOutput();

            Image Infer(const Image& input);
        private:
            void DoAsyncInference();
            void DoInference(const Image& input);

            RadeonRays::thread_safe_queue<Image> m_input_queue;
            RadeonRays::thread_safe_queue<Image> m_output_queue;

            ml_image m_output_image;

            ModelHolder* m_model_holder = nullptr;
            ml_image_info m_input_info;
            ml_image_info m_output_info;

            void Shutdown();

            std::thread m_worker;
            bool m_every_frame = false;
        };
    }
}
