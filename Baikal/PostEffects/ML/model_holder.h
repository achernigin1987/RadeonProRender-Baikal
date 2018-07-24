#pragma once

#include "PostEffects/ML/model.h"

#include <memory>
#include <string>


namespace Baikal
{
    namespace PostEffects
    {
        class ModelHolder
        {
        public:
            ModelHolder();

            ModelHolder(std::string const& model_path,
                        float gpu_memory_fraction,
                        std::string const& visible_devices);

            void reset(std::string const& model_path,
                       float gpu_memory_fraction,
                       std::string const& visible_devices);

            ML::Model* operator ->() const
            {
                return m_model.get();
            }

            ML::Model& operator *() const
            {
                return *m_model;
            }

        private:
            std::unique_ptr<void, void(*)(void*)> m_shared_object;
            std::unique_ptr<ML::Model> m_model;
        };
    }
}
