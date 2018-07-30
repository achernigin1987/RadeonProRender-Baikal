#pragma once

#include "Denoise/denoise_controller.h"

namespace Baikal
{
    class ClwDenoiseController : public DenoiseController {
    public:
        ClwDenoiseController(
                Config* config,
                RenderFactory<ClwScene>::PostEffectType denoiserType,
                size_t width,
                size_t height);

        void SetCamera(Camera::Ptr camera);

    protected:
        void CreateRendererOutputs() override;

    };
}