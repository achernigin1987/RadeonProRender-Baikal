#include "PostEffects/ML/model_holder.h"

#include <stdexcept>

#ifdef WIN32
#   error "Windows is not supported"
#else
#   include <dlfcn.h>
#endif

namespace
{
    constexpr char const* kSharedObject = "libmodel_runner.so";
    constexpr char const* kLoadModelFn = "LoadModel";

    void* LoadSharedObject()
    {
        void* handle = dlopen(kSharedObject, RTLD_NOW);
        if (handle == nullptr)
        {
            throw std::runtime_error(std::string("SO error: ") + dlerror());
        }
        return handle;
    }

    void UnloadSharedObject(void* handle)
    {
        dlclose(handle);
    }
}

namespace Baikal
{
    namespace PostEffects
    {
        ModelHolder::ModelHolder()
            : m_shared_object(LoadSharedObject(), &UnloadSharedObject)
        {
        }

        ModelHolder::ModelHolder(std::string const& model_path,
                                 float gpu_memory_fraction,
                                 std::string const& visible_devices)
            : ModelHolder()
        {
            reset(model_path, gpu_memory_fraction, visible_devices);
        }

        void ModelHolder::reset(std::string const& model_path,
                                float gpu_memory_fraction,
                                std::string const& visible_devices)
        {
            void* symbol = dlsym(m_shared_object.get(), kLoadModelFn);
            if (symbol == nullptr)
            {
                throw std::runtime_error(std::string("Symbol error: ") + dlerror());
            }
            auto load_model = reinterpret_cast<decltype(LoadModel)*>(symbol);

            m_model.reset(load_model(model_path.c_str(),
                                     gpu_memory_fraction,
                                     visible_devices.c_str()));
        }
    }
}
