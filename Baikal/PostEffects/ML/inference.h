#pragma once

#include "buffer.h"


namespace Baikal
{
    namespace PostEffects
    {
        class Inference
        {
        public:
            constexpr size_t INPUT_CHANNELS = 7;
            constexpr size_t OUTPUT_CHANNELS = 3;

            virtual Buffer GetInputBuffer() = 0;
            virtual void PushInput(Buffer&& buffer) = 0;
            virtual Buffer PopOutput() = 0;
            virtual ~Inference() = default;
        };
    }
}
