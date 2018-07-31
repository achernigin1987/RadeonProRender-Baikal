#pragma once

#include "Output/clwoutput.h"
#include "PostEffects/post_effect.h"
#include "RadeonRays/RadeonRays/include/math/float3.h"
#include "RenderFactory/render_factory.h"
#include "Utils/config_manager.h"

#include <memory>
#include <unordered_map>

namespace Baikal
{
    class PostEffectController {
    public:

        PostEffectController(
                Config* config,
                RenderFactory<ClwScene>::PostEffectType type,
                size_t width,
                size_t height);

        RenderFactory<ClwScene>::PostEffectType GetType() const;

        PostEffect::InputTypes GetInputTypes() const;

        RadeonRays::float4 GetParameter(const std::string& name);

        void SetParameter(std::string const& name, RadeonRays::float4 const& value);

        void SetCamera(Camera::Ptr camera);

        void Process(PostEffect::InputSet const& input_set) const;

        Output* GetProcessedOutput() const;

        void Clear() const;

    private:
        using OutputPtr = std::unique_ptr<Baikal::Output>;

        Config* m_config;
        RenderFactory<ClwScene>::PostEffectType m_type;

        std::vector<OutputPtr> m_renderer_outputs;
        OutputPtr m_processed_output;
        Renderer::OutputType m_output_type;

        std::unique_ptr<PostEffect> m_post_effect;
    };
}