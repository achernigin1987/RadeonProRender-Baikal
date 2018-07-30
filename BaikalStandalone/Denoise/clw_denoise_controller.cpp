#include "clw_denoise_controller.h"
#include "PostEffects/bilateral_denoiser.h"
#include "PostEffects/wavelet_denoiser.h"

namespace Baikal
{
    ClwDenoiseController::ClwDenoiseController(
            Config* config,
            RenderFactory<ClwScene>::PostEffectType denoiserType,
            size_t width,
            size_t height)
            : DenoiseController(config, denoiserType, width, height)
    {
        m_denoiser = config->factory->CreatePostEffect(denoiserType);
    }

    void ClwDenoiseController::CreateRendererOutputs()
    {
        // create input set
        AddRendererOutput(Renderer::OutputType::kColor, true);
        AddRendererOutput(Renderer::OutputType::kWorldShadingNormal, true);
        AddRendererOutput(Renderer::OutputType::kWorldPosition, true);
        AddRendererOutput(Renderer::OutputType::kAlbedo, true);
        AddRendererOutput(Renderer::OutputType::kMeshID, true);

        // create denoised output
        AddRendererOutput(Renderer::OutputType::kColor, false);
    }

    void ClwDenoiseController::SetCamera(Camera::Ptr camera)
    {
        if (m_denoiserType == RenderFactory<ClwScene>::PostEffectType::kWaveletDenoiser)
        {
            auto wavelet_denoiser = dynamic_cast<WaveletDenoiser*>(m_denoiser.get());
            wavelet_denoiser->Update(dynamic_cast<PerspectiveCamera*>(camera.get()));
        }
    }
}

