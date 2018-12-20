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

#include "model_holder.h"

#include <stdexcept>
#include <sstream>

namespace Baikal
{
    namespace PostEffects
    {
        ModelHolder::ModelHolder(std::string const& model_path,
                                 float gpu_memory_fraction,
                                 std::string const& visible_devices)
        {
            try
            {
                m_context = mlCreateContext();

                if (m_context == ML_INVALID_HANDLE)
                {
                    throw std::runtime_error("can't create ml context");
                }

                ml_model_params params = {};
                params.model_path = model_path.c_str();
                params.gpu_memory_fraction = gpu_memory_fraction;
                params.visible_devices = !visible_devices.empty() ?
                                         visible_devices.c_str() : nullptr;

                m_model = mlCreateModel(m_context, &params);

                if (m_model == ML_INVALID_HANDLE)
                {
                    throw std::runtime_error(
                            "can't create ml model, check that model is not absent or valid");
                }
            }

            catch (const std::exception& ex)
            {
                ShutDown();
                throw ex;
            }

        }

        ml_image ModelHolder::CreateImage(ml_image_info const& info)
        {
            auto tensor = mlCreateImage(m_context, &info);

            if (tensor == ML_INVALID_HANDLE)
            {
                throw std::runtime_error("can not create model image");
            }

            return tensor;
        }

        ModelHolder::~ModelHolder()
        {
            ShutDown();
        }

        void ModelHolder::ShutDown()
        {
            if (m_model)
            {
                mlReleaseModel(m_model);
            }

            if (m_context)
            {
                mlReleaseContext(m_context);
            }
        }
    }
}
