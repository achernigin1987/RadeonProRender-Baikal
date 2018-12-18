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

#include "PostEffects/ML/model_holder.h"

#include <stdexcept>
#include <sstream>

#ifdef _WIN32
#   include <windows.h>
#else
#   include <dlfcn.h>
#endif


namespace
{
#ifdef _WIN32
    constexpr wchar_t const* kSharedObject = L"model_runner.dll";
#else
    constexpr char const* kSharedObject = "libmodel_runner.so";
#endif
    constexpr char const* kLoadModelFn = "LoadModel";
    decltype(LoadModel)* g_load_model = nullptr;
}

namespace Baikal
{
    namespace PostEffects
    {
        ModelHolder::ModelHolder()
        {
            if (g_load_model != nullptr)
            {
                return;
            }

#ifdef _WIN32
            auto get_last_error = []
            {
                LPSTR buffer = nullptr;
                auto size = FormatMessageA(
                    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                    nullptr,
                    GetLastError(),
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    (LPSTR)&buffer,
                    0,
                    nullptr);
                std::string message(buffer, size);
                LocalFree(buffer);
                return message;
            };

            auto library = LoadLibraryW(kSharedObject);
            if (library == nullptr)
            {
                throw std::runtime_error("Library error: " + get_last_error());
            }

            auto symbol = GetProcAddress(library, kLoadModelFn);
            if (symbol == nullptr)
            {
                throw std::runtime_error(std::string("Symbol error: ") + get_last_error());
            }
#else
            auto library = dlopen(kSharedObject, RTLD_NOW);
            if (library == nullptr)
            {
                throw std::runtime_error(std::string("Library error: ") + dlerror());
            }

            auto symbol = dlsym(library, kLoadModelFn);
            if (symbol == nullptr)
            {
                throw std::runtime_error(std::string("Symbol error: ") + dlerror());
            }
#endif

            g_load_model = reinterpret_cast<decltype(LoadModel)*>(symbol);
        }

        ModelHolder::ModelHolder(std::string const& model_path,
                                 float gpu_memory_fraction,
                                 std::string const& visible_devices)
        : ModelHolder()
        {
            Reset(model_path, gpu_memory_fraction, visible_devices);
        }

        void ModelHolder::Reset(std::string const& model_path,
                                float gpu_memory_fraction,
                                std::string const& visible_devices)
        {
            ModelParams params = {};
            params.model_path = model_path.c_str();
            params.gpu_memory_fraction = gpu_memory_fraction;
            params.visible_devices = visible_devices.c_str();
            m_model.reset(g_load_model(&params));
        }
    }
}
