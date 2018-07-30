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
    class DenoiseController {
    public:
        DenoiseController(
                Config* config,
                RenderFactory<ClwScene>::PostEffectType denoiserType,
                size_t width,
                size_t height);

        void Init();

        RenderFactory<ClwScene>::PostEffectType GetType() const;

        RadeonRays::float4 GetParameter(const std::string& name);

        void SetParameter(std::string const& name, RadeonRays::float4 const& value);

        void Process() const;

        void GetProcessedData(RadeonRays::float3* data) const;

        void Clear() const;

        //void RestoreDenoiserOutput(Renderer::OutputType type) const;

    protected:
        using OutputPtr = std::unique_ptr<Baikal::Output>;

        virtual void CreateRendererOutputs() = 0;

        void AddRendererOutput(Renderer::OutputType type, bool add_to_input);

        Config* m_config;
        RenderFactory<ClwScene>::PostEffectType m_denoiserType;

        std::vector<OutputPtr> m_outputs;

        size_t m_width;
        size_t m_height;

        std::unique_ptr<PostEffect> m_denoiser;
    private:
        PostEffect::InputSet m_input_set;
    };
}