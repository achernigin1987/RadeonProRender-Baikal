
/**********************************************************************
 Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ********************************************************************/
#pragma once

#include "RenderFactory/render_factory.h"
#include "Renderers/monte_carlo_renderer.h"
#include "Output/clwoutput.h"
#include "Application/app_utils.h"
#include "Utils/config_manager.h"
#include "Application/gl_render.h"
#include "SceneGraph/camera.h"

#ifdef ENABLE_DENOISER
#include "PostEffects/post_effect.h"
#endif

#include <thread>
#include <atomic>
#include <mutex>
#include <future>


namespace Baikal
{
    class AppClRender
    {
        struct OutputData
        {
            std::unique_ptr<Baikal::Output> tmp_output;
            std::vector<float3> fdata;
            std::vector<unsigned char> udata;
            CLWBuffer<float3> copybuffer;
        };

        struct ControlData
        {
            std::atomic<int> clear;
            std::atomic<int> stop;
            std::atomic<int> newdata;
            std::mutex datamutex;
            int idx;
        };

    public:
        AppClRender(AppSettings& settings, GLuint tex);
        //copy data from to GL
        void Update(AppSettings& settings);

        //compile scene
        void UpdateScene();
        //render
        void Render(int sample_cnt);
        void StartRenderThreads();
        void StopRenderThreads();
        void RunBenchmark(AppSettings& settings);

        //save cl frame buffer to file
        void SaveFrameBuffer(AppSettings& settings);
        void SaveImage(const std::string& name, int width, int height, const RadeonRays::float3* data);

        inline Baikal::Camera::Ptr GetCamera() { return m_camera; };
        inline Baikal::Scene1::Ptr GetScene() { return m_scene; };
        inline CLWDevice GetDevice(int i) { return m_cfgs[m_primary].context.GetDevice(i); };
        inline Renderer::OutputType GetOutputType() { return m_output_type; };

        void SetNumBounces(int num_bounces);
        void SetOutputType(Renderer::OutputType type);

        std::future<int> GetShapeId(std::uint32_t x, std::uint32_t y);
        Baikal::Shape::Ptr GetShapeById(int shape_id);

#ifdef ENABLE_DENOISER
        PostEffect::Type GetPostEffectType() const;
        void SetDenoiserFloatParam(const std::string& name, const float4& value);
        float4 GetDenoiserFloatParam(const std::string& name);
        void RestoreDenoiserOutput(std::size_t cfg_index, Renderer::OutputType type) const;
#endif

    private:
        using RendererOutputs = std::map<Renderer::OutputType, std::unique_ptr<Output>>;

        void InitCl(AppSettings& settings, GLuint tex);
        void LoadScene(AppSettings& settings);
        void RenderThread(ControlData& cd);

        Output* GetRendererOutput(size_t device_idx, Renderer::OutputType type);
        void AddRendererOutput(size_t device_idx, Renderer::OutputType type);
        void GetOutputData(size_t device_idx, Renderer::OutputType type, RadeonRays::float3* data) const;
        void AddPostEffect(size_t device_idx, PostEffect::Type type);

        void ApplyGammaCorrection(size_t device_idx);

        Baikal::Scene1::Ptr m_scene;
        Baikal::Camera::Ptr m_camera;

        std::promise<int> m_promise;
        bool m_shape_id_requested = false;
        OutputData m_shape_id_data;
        OutputData m_dummy_output_data;
        RadeonRays::float2 m_shape_id_pos;
        std::vector<Config> m_cfgs;
        std::vector<RendererOutputs> m_renderer_outputs;

        std::vector<OutputData> m_outputs;
        std::unique_ptr<ControlData[]> m_ctrl;
        std::vector<std::thread> m_renderthreads;
        size_t m_primary;
        std::uint32_t m_width, m_height;

        //if interop
        CLWImage2D m_cl_interop_image;
        //save GL tex for no interop case
        GLuint m_tex;
        Renderer::OutputType m_output_type;

#ifdef ENABLE_DENOISER
        std::unique_ptr<PostEffect> m_post_effect;
        PostEffect::Type m_post_effect_type;
        PostEffect::InputSet m_post_effect_inputs;
        std::unique_ptr<Output> m_post_effect_output;
#endif
    };
}
