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

        RadeonRays::float4 GetParameter(const std::string& name);

        void SetParameter(std::string const& name, RadeonRays::float4 const& value);

        void SetCamera(Camera::Ptr camera);

        void Process() const;

        void GetProcessedData(RadeonRays::float3* data) const;

        Output* GetProcessedOutput() const;

        void Clear() const;

        //void RestoreDenoiserOutput(Renderer::OutputType type) const;

    protected:
        using OutputPtr = std::unique_ptr<Baikal::Output>;

        void CreateRendererOutputs();

        void AddRendererOutput(Renderer::OutputType type, bool add_to_input);

        Config* m_config;
        RenderFactory<ClwScene>::PostEffectType m_type;

        std::vector<OutputPtr> m_outputs;

        size_t m_width;
        size_t m_height;

        std::unique_ptr<PostEffect> m_post_effect;
    private:
        PostEffect::InputSet m_input_set;
    };
}