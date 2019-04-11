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

#include "PostEffects/ML/ml_common.h"
#include "RadeonProML.h"
#include "CLW.h"

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
            ModelHolder(ModelType model_type,
                        std::string const& model_path,
                        float gpu_memory_fraction,
                        std::string const& visible_devices,
                        cl_command_queue command_queue = nullptr);

            ModelHolder(const ModelHolder&) = delete;
            ModelHolder(ModelHolder&&) = delete;

            ModelHolder& operator = (const ModelHolder&) = delete;
            ModelHolder& operator = (ModelHolder&&) = delete;

            ~ModelHolder();

            const ml_context GetContext() const { return m_context; }
            const ml_model GetModel() const { return m_model; }
            const ModelType GetModelType() const { return m_model_type; }

            ml_image CreateImage(ml_image_info const& info, ml_access_mode access_mode);
            ml_image CreateImageFromBuffer(cl_mem buffer, ml_image_info const& info, ml_access_mode access_mode);

        private:
            const ModelType m_model_type;
            ml_context m_context;
            ml_model m_model;
        };
    }
}
