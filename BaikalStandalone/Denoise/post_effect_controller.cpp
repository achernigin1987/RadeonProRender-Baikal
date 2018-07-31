#include "Denoise/post_effect_controller.h"
#include "PostEffects/bilateral_denoiser.h"
#include "PostEffects/wavelet_denoiser.h"
#include "PostEffects/ML/denoiser.h"


namespace Baikal
{
    PostEffectController::PostEffectController(
            Config* config,
            RenderFactory<ClwScene>::PostEffectType type,
            size_t width,
            size_t height)
    : m_config(config)
    , m_type(type)
    , m_post_effect(config->factory->CreatePostEffect(type, width, height))
    {
        // create output for post-effect result
        m_processed_output = m_config->factory->CreateOutput(
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height));
    }

    RenderFactory<ClwScene>::PostEffectType PostEffectController::GetType() const
    {
        return m_type;
    }

    PostEffect::InputTypes PostEffectController::GetInputTypes() const
    {
        return m_post_effect->GetInputTypes();
    }

    void PostEffectController::SetParameter(std::string const& name, RadeonRays::float4 const& value)
    {
        m_post_effect->SetParameter(name, value);
    }

    RadeonRays::float4 PostEffectController::GetParameter(const std::string& name)
    {
        return m_post_effect->GetParameter(name);
    }

    void PostEffectController::SetCamera(Camera::Ptr camera)
    {
        if (m_type == RenderFactory<ClwScene>::PostEffectType::kWaveletDenoiser)
        {
            auto wavelet_denoiser = dynamic_cast<WaveletDenoiser*>(m_post_effect.get());
            wavelet_denoiser->Update(dynamic_cast<PerspectiveCamera*>(camera.get()));
        }
    }

    Output* PostEffectController::GetProcessedOutput() const
    {
        return m_processed_output.get();
    }

    void PostEffectController::Clear() const
    {
        m_config->renderer->Clear(float3(0, 0, 0), *m_processed_output);
    }

    void PostEffectController::Process(PostEffect::InputSet const& input_set) const
    {
        m_post_effect->Apply(input_set, *m_processed_output);
    }
}