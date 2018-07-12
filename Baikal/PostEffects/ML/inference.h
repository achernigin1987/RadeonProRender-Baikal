#pragma once

#include "buffer.h"


namespace Baikal::ML
{
    namespace PostEffects
    {
        class Inference
        {
        public:
            virtual Buffer GetInputBuffer() = 0;
            virtual void PushInput(Buffer&& buffer) = 0;
            virtual Buffer PopOutput() = 0;
            virtual ~Inference() = default;
        };
    }
}
