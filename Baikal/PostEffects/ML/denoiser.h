#pragma once

#include "Baikal/PostEffects/ML/inference.h"


namespace Baikal
{
    namespace PostEffects
    {
        enum class DenoiseInputs
        {
            kColorDepthNormalGloss7,
        };

        std::unique_ptr<Inference> CreateDenoiser(size_t width,
                                                  size_t height,
                                                  DenoiseInputs inputs);
    }
}
