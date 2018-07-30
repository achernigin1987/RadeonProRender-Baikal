#include "denoise_controller.h"

#include "PostEffects/ML/denoiser.h"


namespace Baikal
{
    DenoiseController::DenoiseController(
            Config* config,
            RenderFactory<ClwScene>::PostEffectType denoiserType,
            size_t width,
            size_t height)
    : m_config(config)
    , m_denoiserType(denoiserType)
    , m_width(width)
    , m_height(height)
    {

    }

    void DenoiseController::Init()
    {
        CreateRendererOutputs();
    }

    RenderFactory<ClwScene>::PostEffectType DenoiseController::GetType() const
    {
        return m_denoiserType;
    }

    void DenoiseController::SetParameter(std::string const& name, RadeonRays::float4 const& value)
    {
        m_denoiser->SetParameter(name, value);
    }

    RadeonRays::float4 DenoiseController::GetParameter(const std::string& name)
    {
        return m_denoiser->GetParameter(name);
    }

    void DenoiseController::GetProcessedData(RadeonRays::float3* data) const
    {
        auto processed_output = m_outputs.back().get();
        processed_output->GetData(data);
    }

    void DenoiseController::AddRendererOutput(Renderer::OutputType type, bool add_to_input)
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

    void DenoiseController::Clear() const
    {
        for (const auto& output : m_outputs)
        {
            m_config->renderer->Clear(float3(0, 0, 0), *output);
        }
    }

    void DenoiseController::Process() const
    {
        auto processed_output = m_outputs.back().get();
        m_denoiser->Apply(m_input_set, *processed_output);
    }

//    void DenoiseController::RestoreDenoiserOutput(Renderer::OutputType type) const
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