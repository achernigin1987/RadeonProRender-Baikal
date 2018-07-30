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
    , m_width(width)
    , m_height(height)
    , m_post_effect(config->factory->CreatePostEffect(type, m_width, m_height))
    {
        CreateRendererOutputs();
    }

    RenderFactory<ClwScene>::PostEffectType PostEffectController::GetType() const
    {
        return m_type;
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

    void PostEffectController::GetProcessedData(RadeonRays::float3* data) const
    {
        GetProcessedOutput()->GetData(data);
    }

    Output* PostEffectController::GetProcessedOutput() const
    {
        return m_outputs.back().get();
    }

    void PostEffectController::CreateRendererOutputs()
    {
        // create input set
        switch (m_type)
        {
            case RenderFactory<ClwScene>::PostEffectType::kBilateralDenoiser:
            case RenderFactory<ClwScene>::PostEffectType::kWaveletDenoiser:

                AddRendererOutput(Renderer::OutputType::kColor, true);
                AddRendererOutput(Renderer::OutputType::kWorldShadingNormal, true);
                AddRendererOutput(Renderer::OutputType::kWorldPosition, true);
                AddRendererOutput(Renderer::OutputType::kAlbedo, true);
                AddRendererOutput(Renderer::OutputType::kMeshID, true);
                break;
            case RenderFactory<ClwScene>::PostEffectType::kMLDenoiser:
                AddRendererOutput(Renderer::OutputType::kColor, true);
                AddRendererOutput(Renderer::OutputType::kAlbedo, true);
                AddRendererOutput(Renderer::OutputType::kViewShadingNormal, true);
                break;
        }
        // create denoised output
        AddRendererOutput(Renderer::OutputType::kColor, false);
    }

    void PostEffectController::AddRendererOutput(Renderer::OutputType type, bool add_to_input)
    {
        auto output = m_config->factory->CreateOutput(static_cast<std::uint32_t>(m_width),
                                                      static_cast<std::uint32_t>(m_height));
        m_config->renderer->SetOutput(type, output.get());
        if (add_to_input)
        {
            m_input_set[type] = output.get();
        }
        m_outputs.push_back(std::move(output));
    }

    void PostEffectController::Clear() const
    {
        for (const auto& output : m_outputs)
        {
            m_config->renderer->Clear(float3(0, 0, 0), *output);
        }
    }

    void PostEffectController::Process() const
    {
        m_post_effect->Apply(m_input_set, *GetProcessedOutput());
    }

//    void PostEffectController::RestoreDenoiserOutput(Renderer::OutputType type) const
//    {
//        switch (type)
//        {
//            case Renderer::OutputType::kWorldShadingNormal:
//                m_config->renderer->SetOutput(type, output_normal.get());
//                break;
//            case Renderer::OutputType::kWorldPosition:
//                m_config->renderer->SetOutput(type, output_position.get());
//                break;
//            case Renderer::OutputType::kAlbedo:
//                m_config->renderer->SetOutput(type, output_albedo.get());
//                break;
//            case Renderer::OutputType::kMeshID:
//                m_config->renderer->SetOutput(type, output_mesh_id.get());
//                break;
//            default:
//                // Nothing to restore
//                m_config->renderer->SetOutput(type, nullptr);
//                break;
//        }
//    }
}