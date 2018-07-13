#include "Baikal/PostEffects/ML/denoiser.h"

#include "Baikal/PostEffects/ML/inference_impl.h"


namespace Baikal 
{
    namespace PostEffects 
    {
        std::unique_ptr<Inference> Baikal::PostEffects::CreateDenoiser(size_t width,
                                                                       size_t height,
                                                                       DenoiseInputs inputs)
        {
            return std::make_unique<InferenceImpl>("model/path.pb", width, height);
        }
    }
}
