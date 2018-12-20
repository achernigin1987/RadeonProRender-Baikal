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

#include <ml.h>

#include <memory>
#include <string>


namespace Baikal
{
    namespace PostEffects
    {
        // non copyable/movable
        class ModelHolder
        {
        public:
            ModelHolder(std::string const& model_path,
                        float gpu_memory_fraction,
                        std::string const& visible_devices);


            ml_model GetModel()
            {
                return m_model;
            }

            ml_image CreateImage(ml_image_info const& info);

            ~ModelHolder();


            ModelHolder(const ModelHolder&) = delete;
            ModelHolder(ModelHolder&&) = delete;
            ModelHolder& operator = (const ModelHolder&) = delete;
            ModelHolder& operator = (ModelHolder&&) = delete;

        private:
            void ShutDown();

            ml_model m_model;
            ml_context m_context;
        };
    }
}
