/**********************************************************************
Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.

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
#include "Application/cl_render.h"
#include "Application/gl_render.h"

#include "SceneGraph/scene1.h"
#include "SceneGraph/camera.h"
#include "SceneGraph/material.h"
#include "BaikalIO/scene_io.h"
#include "BaikalIO/image_io.h"
#include "BaikalIO/material_io.h"
#include "Renderers/monte_carlo_renderer.h"
#include "Renderers/adaptive_renderer.h"
#include "Utils/clw_class.h"

#ifdef ENABLE_DENOISER
#include "PostEffects/wavelet_denoiser.h"
#endif

#include "OpenImageIO/imageio.h"

#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <cmath>


namespace Baikal
{
    AppClRender::AppClRender(AppSettings& settings, GLuint tex)
    : m_tex(tex), m_output_type(Renderer::OutputType::kColor)
    {
        InitCl(settings, m_tex);

#ifdef ENABLE_DENOISER
        AddPostEffect(m_primary, PostEffect::Type::kWaveletDenoiser);
#endif

        LoadScene(settings);
    }

    void AppClRender::InitCl(AppSettings& settings, GLuint tex)
    {
        bool force_disable_interop = false;
        //create cl context
        try
        {
            CreateConfigs(
                settings.mode,
                settings.interop,
                m_cfgs,
                settings.num_bounces,
                settings.platform_index,
                settings.device_index);
        }
        catch (CLWException &)
        {
            force_disable_interop = true;
            CreateConfigs(settings.mode, false, m_cfgs, settings.num_bounces, settings.platform_index, settings.device_index);
        }

        m_width = (std::uint32_t)settings.width;
        m_height = (std::uint32_t)settings.height;

        std::cout << "Running on devices: \n";

        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            std::cout << i << ": " << m_cfgs[i].context.GetDevice(0).GetName() << "\n";
        }

        settings.interop = false;

        m_outputs.resize(m_cfgs.size());
        m_ctrl.reset(new ControlData[m_cfgs.size()]);

        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            if (m_cfgs[i].type == DeviceType::kPrimary)
            {
                m_primary = i;

                if (m_cfgs[i].caninterop)
                {
                    m_cl_interop_image = m_cfgs[i].context.CreateImage2DFromGLTexture(tex);
                    settings.interop = true;
                }
            }

            m_ctrl[i].clear.store(1);
            m_ctrl[i].stop.store(0);
            m_ctrl[i].newdata.store(0);
            m_ctrl[i].idx = static_cast<int>(i);
        }

        if (force_disable_interop)
        {
            std::cout << "OpenGL interop is not supported, disabled, -interop flag is ignored\n";
        }
        else
        {
            if (settings.interop)
            {
                std::cout << "OpenGL interop mode enabled\n";
            }
            else
            {
                std::cout << "OpenGL interop mode disabled\n";
            }
        }

        m_renderer_outputs.resize(m_cfgs.size());
        //create renderer
        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            AddRendererOutput(i, Baikal::Renderer::OutputType::kColor);

            m_outputs[i].fdata.resize(settings.width * settings.height);
            m_outputs[i].udata.resize(settings.width * settings.height * 4);

            if (m_cfgs[i].type == DeviceType::kPrimary)
            {
                m_outputs[i].copybuffer = m_cfgs[i].context.CreateBuffer<RadeonRays::float3>(m_width * m_height, CL_MEM_READ_WRITE);
            }
        }

        m_shape_id_data.tmp_output = m_cfgs[m_primary].factory->CreateOutput(m_width, m_height);
        m_dummy_output_data.tmp_output = m_cfgs[m_primary].factory->CreateOutput(m_width, m_height);

        m_cfgs[m_primary].renderer->Clear(RadeonRays::float3(0, 0, 0), *m_shape_id_data.tmp_output);
        m_cfgs[m_primary].renderer->Clear(RadeonRays::float3(0, 0, 0), *m_dummy_output_data.tmp_output);
    }

    void AppClRender::AddRendererOutput(size_t device_idx, Renderer::OutputType type)
    {
        auto it = m_renderer_outputs[device_idx].find(type);
        if (it == m_renderer_outputs[device_idx].end())
        {
            const auto& config = m_cfgs.at(device_idx);
            auto output = config.factory->CreateOutput(m_width, m_height);
            config.renderer->SetOutput(type, output.get());
            config.renderer->Clear(RadeonRays::float3(0, 0, 0), *output);

            m_renderer_outputs[device_idx].emplace(type, std::move(output));
        }
    }

    Output* AppClRender::GetRendererOutput(size_t device_idx, Renderer::OutputType type)
    {
        return m_renderer_outputs.at(device_idx).at(type).get();
    }

    void AppClRender::GetOutputData(size_t device_idx, Renderer::OutputType type, RadeonRays::float3* data) const
    {
        m_renderer_outputs.at(device_idx).at(type)->GetData(data);
    }

#ifdef ENABLE_DENOISER
    void AppClRender::AddPostEffect(size_t device_idx, PostEffect::Type type)
    {
        m_post_effect_type = type;

        m_post_effect = m_cfgs[device_idx].factory->CreatePostEffect(type, m_width, m_height);

        // create or get inputs for post-effect
        for (auto required_input : m_post_effect->GetInputTypes())
        {
            AddRendererOutput(device_idx, required_input);
            m_post_effect_inputs[required_input] = GetRendererOutput(device_idx, required_input);
        }

        // create buffer for post-effect output
        m_post_effect_output = m_cfgs[device_idx].factory->CreateOutput(m_width, m_height);
    }

    PostEffect::Type AppClRender::GetPostEffectType() const
    {
        return m_post_effect_type;
    }

    void AppClRender::SetDenoiserFloatParam(const std::string& name, const float4& value)
    {
        m_post_effect->SetParameter(name, value);
    }

    float4 AppClRender::GetDenoiserFloatParam(const std::string& name)
    {
        return m_post_effect->GetParameter(name);
    }
#endif

    void AppClRender::LoadScene(AppSettings& settings)
    {
        rand_init();

        // Load obj file
        std::string basepath = settings.path;
        basepath += "/";
        std::string filename = basepath + settings.modelname;

        {
            m_scene = Baikal::SceneIo::LoadScene(filename, basepath);

            {
            #ifdef WIN32
            #undef LoadImage
            #endif
                auto image_io(ImageIo::CreateImageIo());
                auto ibl_texture = image_io->LoadImage(settings.envmapname);

                auto ibl = ImageBasedLight::Create();
                ibl->SetTexture(ibl_texture);
                m_scene->AttachLight(ibl);
            }

            // Enable this to generate new materal mapping for a model
#if 0
            auto material_io{Baikal::MaterialIo::CreateMaterialIoXML()};
            material_io->SaveMaterialsFromScene(basepath + "materials.xml", *m_scene);
            material_io->SaveIdentityMapping(basepath + "mapping.xml", *m_scene);
#endif

            // Check it we have material remapping
            std::ifstream in_materials(basepath + "materials.xml");
            std::ifstream in_mapping(basepath + "mapping.xml");

            if (in_materials && in_mapping)
            {
                in_materials.close();
                in_mapping.close();

                auto material_io = Baikal::MaterialIo::CreateMaterialIoXML();
                auto mats = material_io->LoadMaterials(basepath + "materials.xml");
                auto mapping = material_io->LoadMaterialMapping(basepath + "mapping.xml");

                material_io->ReplaceSceneMaterials(*m_scene, *mats, mapping);
            }
        }

        switch (settings.camera_type)
        {
        case CameraType::kPerspective:
            m_camera = Baikal::PerspectiveCamera::Create(
                settings.camera_pos
                , settings.camera_at
                , settings.camera_up);

            break;
        case CameraType::kOrthographic:
            m_camera = Baikal::OrthographicCamera::Create(
                settings.camera_pos
                , settings.camera_at
                , settings.camera_up);
            break;
        default:
            throw std::runtime_error("AppClRender::InitCl(...): unsupported camera type");
        }

        m_scene->SetCamera(m_camera);

        // Adjust sensor size based on current aspect ratio
        float aspect = (float)settings.width / settings.height;
        settings.camera_sensor_size.y = settings.camera_sensor_size.x / aspect;

        m_camera->SetSensorSize(settings.camera_sensor_size);
        m_camera->SetDepthRange(settings.camera_zcap);

        auto perspective_camera = std::dynamic_pointer_cast<Baikal::PerspectiveCamera>(m_camera);

        // if camera mode is kPerspective
        if (perspective_camera)
        {
            perspective_camera->SetFocalLength(settings.camera_focal_length);
            perspective_camera->SetFocusDistance(settings.camera_focus_distance);
            perspective_camera->SetAperture(settings.camera_aperture);
            std::cout << "Camera type: " << (perspective_camera->GetAperture() > 0.f ? "Physical" : "Pinhole") << "\n";
            std::cout << "Lens focal length: " << perspective_camera->GetFocalLength() * 1000.f << "mm\n";
            std::cout << "Lens focus distance: " << perspective_camera->GetFocusDistance() << "m\n";
            std::cout << "F-Stop: " << 1.f / (perspective_camera->GetAperture() * 10.f) << "\n";
        }

        std::cout << "Sensor size: " << settings.camera_sensor_size.x * 1000.f << "x" << settings.camera_sensor_size.y * 1000.f << "mm\n";
    }

    void AppClRender::UpdateScene()
    {
        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            if (i == m_primary)
            {
                m_cfgs[i].controller->CompileScene(m_scene);
                m_cfgs[i].renderer->Clear(float3(), *GetRendererOutput(i, Renderer::OutputType::kColor));

#ifdef ENABLE_DENOISER
                m_post_effect_output->Clear(float3());
#endif
            }
            else
                m_ctrl[i].clear.store(true);
        }
    }

    void AppClRender::Update(AppSettings& settings)
    {
        //if (std::chrono::duration_cast<std::chrono::seconds>(time - updatetime).count() > 1)
        //{
        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            if (m_cfgs[i].type == DeviceType::kPrimary)
                continue;

            int desired = 1;
            if (std::atomic_compare_exchange_strong(&m_ctrl[i].newdata, &desired, 0))
            {
                {
                    m_cfgs[m_primary].context.WriteBuffer(
                            0, m_outputs[m_primary].copybuffer,
                            &m_outputs[i].fdata[0],
                            settings.width * settings.height);
                }

                auto acckernel = static_cast<MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetAccumulateKernel();

                int argc = 0;
                acckernel.SetArg(argc++, m_outputs[m_primary].copybuffer);
                acckernel.SetArg(argc++, settings.width * settings.width);
                acckernel.SetArg(argc++,
                        static_cast<Baikal::ClwOutput*>(GetRendererOutput(
                                m_primary, Renderer::OutputType::kColor))->data());

                int globalsize = settings.width * settings.height;
                m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, acckernel);
            }
        }

        //updatetime = time;
        //}

        if (!settings.interop)
        {
#ifdef ENABLE_DENOISER
            m_post_effect_output->GetData(&m_outputs[m_primary].fdata[0]);
#else
            GetOutputData(m_primary, Renderer::OutputType::kColor, &m_renderer_outputs[m_primary].fdata[0]);
#endif
            ApplyGammaCorrection(m_primary);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, m_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, &m_outputs[m_primary].udata[0]);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        else
        {
            std::vector<cl_mem> objects;
            objects.push_back(m_cl_interop_image);
            m_cfgs[m_primary].context.AcquireGLObjects(0, objects);

            auto copykernel = static_cast<MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->GetCopyKernel();

#ifdef ENABLE_DENOISER
            auto output = m_post_effect_output.get();
#else
            auto output = GetRendererOutput(m_primary, Renderer::OutputType::kColor);
#endif

            unsigned int argc = 0;

            copykernel.SetArg(argc++, static_cast<Baikal::ClwOutput*>(output)->data());
            copykernel.SetArg(argc++, output->width());
            copykernel.SetArg(argc++, output->height());
            copykernel.SetArg(argc++, 2.2f);
            copykernel.SetArg(argc++, m_cl_interop_image);

            int globalsize = output->width() * output->height();
            m_cfgs[m_primary].context.Launch1D(0, ((globalsize + 63) / 64) * 64, 64, copykernel);

            m_cfgs[m_primary].context.ReleaseGLObjects(0, objects);
            m_cfgs[m_primary].context.Finish(0);
        }


        if (settings.benchmark)
        {
            auto& scene = m_cfgs[m_primary].controller->CompileScene(m_scene);
            static_cast<MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->Benchmark(scene, settings.stats);

            settings.benchmark = false;
            settings.rt_benchmarked = true;
        }

        //ClwClass::Update();
    }

    void AppClRender::Render(int samples)
    {
#ifdef ENABLE_DENOISER
        m_post_effect->Update(m_camera.get(), static_cast<unsigned>(samples));
#endif
        auto& scene = m_cfgs[m_primary].controller->GetCachedScene(m_scene);
        m_cfgs[m_primary].renderer->Render(scene);

        if (m_shape_id_requested)
        {
            // offset in OpenCl memory till necessary item
            auto offset = (std::uint32_t)(m_width * (m_height - m_shape_id_pos.y) + m_shape_id_pos.x);
            // copy shape id elem from OpenCl
            float4 shape_id;
            m_shape_id_data.tmp_output->GetData((float3*)&shape_id, offset, 1);
            m_promise.set_value(static_cast<int>(shape_id.x));
            // clear output to stop tracking shape id map in openCl
            m_cfgs[m_primary].renderer->SetOutput(Renderer::OutputType::kShapeId, nullptr);
            m_shape_id_requested = false;
        }

#ifdef ENABLE_DENOISER
        m_post_effect->Apply(m_post_effect_inputs, *m_post_effect_output);
#endif
    }

    void AppClRender::SaveFrameBuffer(AppSettings& settings)
    {
        std::vector<RadeonRays::float3> data;

        //read cl output in case of iterop
        std::vector<RadeonRays::float3> output_data;
        if (settings.interop)
        {
            auto output = GetRendererOutput(m_primary, Renderer::OutputType::kColor);
            auto buffer = static_cast<Baikal::ClwOutput*>(output)->data();
            output_data.resize(buffer.GetElementCount());
            m_cfgs[m_primary].context.ReadBuffer(
                    0,
                    static_cast<Baikal::ClwOutput*>(output)->data(),
                    &output_data[0],
                    output_data.size()).Wait();
        }

        //use already copied to CPU cl data in case of no interop
        auto& fdata = settings.interop ? output_data : m_outputs[m_primary].fdata;

        data.resize(fdata.size());
        std::transform(fdata.cbegin(), fdata.cend(), data.begin(),
            [](RadeonRays::float3 const& v)
        {
            float invw = 1.f / v.w;
            return v * invw;
        });

        std::stringstream oss;
        auto camera_position = m_camera->GetPosition();
        auto camera_direction = m_camera->GetForwardVector();
        oss << "../Output/" << settings.modelname << "_p" << camera_position.x << camera_position.y << camera_position.z <<
            "_d" << camera_direction.x << camera_direction.y << camera_direction.z <<
            "_s" << settings.num_samples << ".exr";

        SaveImage(oss.str(), settings.width, settings.height, data.data());
    }

    void AppClRender::SaveImage(const std::string& name, int width, int height, const RadeonRays::float3* data)
    {
        OIIO_NAMESPACE_USING;

        std::vector<float3> tempbuf(width * height);
        tempbuf.assign(data, data + width * height);

        for (auto y = 0; y < height; ++y)
            for (auto x = 0; x < width; ++x)
            {

                float3 val = data[(height - 1 - y) * width + x];
                tempbuf[y * width + x] = (1.f / val.w) * val;

                tempbuf[y * width + x].x = std::pow(tempbuf[y * width + x].x, 1.f / 2.2f);
                tempbuf[y * width + x].y = std::pow(tempbuf[y * width + x].y, 1.f / 2.2f);
                tempbuf[y * width + x].z = std::pow(tempbuf[y * width + x].z, 1.f / 2.2f);
            }

        ImageOutput* out = ImageOutput::create(name);

        if (!out)
        {
            throw std::runtime_error("Can't create image file on disk");
        }

        ImageSpec spec(width, height, 3, TypeDesc::FLOAT);

        out->open(name, spec);
        out->write_image(TypeDesc::FLOAT, &tempbuf[0], sizeof(float3));
        out->close();
    }

    void AppClRender::RenderThread(ControlData& cd)
    {
        auto renderer = m_cfgs[cd.idx].renderer.get();
        auto controller = m_cfgs[cd.idx].controller.get();
        auto output = GetRendererOutput(m_primary, Renderer::OutputType::kColor);

        auto updatetime = std::chrono::high_resolution_clock::now();

        while (!cd.stop.load())
        {
            int result = 1;
            bool update = false;

            if (std::atomic_compare_exchange_strong(&cd.clear, &result, 0))
            {
                renderer->Clear(float3(0, 0, 0), *output);
                controller->CompileScene(m_scene);
                update = true;
            }

            auto& scene = controller->GetCachedScene(m_scene);
            renderer->Render(scene);

            auto now = std::chrono::high_resolution_clock::now();

            update = update || (std::chrono::duration_cast<std::chrono::seconds>(now - updatetime).count() > 1);

            if (update)
            {
                GetOutputData(cd.idx, Renderer::OutputType::kColor, &m_outputs[cd.idx].fdata[0]);
                updatetime = now;
                cd.newdata.store(1);
            }

            m_cfgs[cd.idx].context.Finish(0);
        }
    }

    void AppClRender::StartRenderThreads()
    {
        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            if (i != m_primary)
            {
                m_renderthreads.push_back(std::thread(&AppClRender::RenderThread, this, std::ref(m_ctrl[i])));
                m_renderthreads.back().detach();
            }
        }

        std::cout << m_cfgs.size() << " OpenCL submission threads started\n";
    }

    void AppClRender::StopRenderThreads()
    {
        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            if (i != m_primary)
            {
                m_ctrl[i].stop.store(true);
            }
        }
    }

    void AppClRender::RunBenchmark(AppSettings& settings)
    {
        std::cout << "Running general benchmark...\n";

        auto time_bench_start_time = std::chrono::high_resolution_clock::now();
        for (auto i = 0U; i < 512; ++i)
        {
            Render(0);
        }

        m_cfgs[m_primary].context.Finish(0);

        auto delta = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now() - time_bench_start_time).count();

        settings.time_benchmark_time = delta / 1000.f;

        GetOutputData(m_primary, Renderer::OutputType::kColor, &m_outputs[m_primary].fdata[0]);
        ApplyGammaCorrection(m_primary);

        auto& fdata = m_outputs[m_primary].fdata;
        std::vector<RadeonRays::float3> data(fdata.size());
        std::transform(fdata.cbegin(), fdata.cend(), data.begin(),
            [](RadeonRays::float3 const& v)
        {
            float invw = 1.f / v.w;
            return v * invw;
        });

        std::stringstream oss;
        oss << "../Output/" << settings.modelname << ".exr";

        SaveImage(oss.str(), settings.width, settings.height, &data[0]);

        std::cout << "Running RT benchmark...\n";

        auto& scene = m_cfgs[m_primary].controller->GetCachedScene(m_scene);
        static_cast<MonteCarloRenderer*>(m_cfgs[m_primary].renderer.get())->Benchmark(scene, settings.stats);
    }

    void AppClRender::ApplyGammaCorrection(size_t device_idx)
    {
        auto adjust_gamma = [](float val)
        {
            return (unsigned char)clamp(std::pow(val , 1.f / 2.2f) * 255, 0, 255);
        };

        for (int i = 0; i < (int)m_outputs[device_idx].fdata.size(); ++i)
        {
            auto &fdata = m_outputs[device_idx].fdata[i];
            auto &udata = m_outputs[device_idx].udata;

            udata[4 * i] = adjust_gamma(fdata.x / fdata.w);
            udata[4 * i + 1] = adjust_gamma(fdata.y / fdata.w);
            udata[4 * i + 2] = adjust_gamma(fdata.z / fdata.w);
            udata[4 * i + 3] = 1;
        }
    }

    void AppClRender::SetNumBounces(int num_bounces)
    {
        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
            static_cast<Baikal::MonteCarloRenderer*>(m_cfgs[i].renderer.get())->SetMaxBounces(num_bounces);
        }
    }

    void AppClRender::SetOutputType(Renderer::OutputType type)
    {
        for (std::size_t i = 0; i < m_cfgs.size(); ++i)
        {
#ifdef ENABLE_DENOISER
            RestoreDenoiserOutput(i, m_output_type);
#else
            m_cfgs[i].renderer->SetOutput(m_output_type, nullptr);
#endif
            if (type == Renderer::OutputType::kOpacity || type == Renderer::OutputType::kVisibility)
            {
                m_cfgs[i].renderer->SetOutput(Renderer::OutputType::kColor, m_dummy_output_data.tmp_output.get());
            }
            else
            {
                m_cfgs[i].renderer->SetOutput(Renderer::OutputType::kColor, nullptr);
            }
            m_cfgs[i].renderer->SetOutput(type, GetRendererOutput(i, Renderer::OutputType::kColor));
        }
        m_output_type = type;
    }


    std::future<int> AppClRender::GetShapeId(std::uint32_t x, std::uint32_t y)
    {
        m_promise = std::promise<int>();
        if (x >= m_width || y >= m_height)
            throw std::logic_error(
                "AppClRender::GetShapeId(...): x or y cords beyond the size of image");

        if (m_cfgs.empty())
            throw std::runtime_error("AppClRender::GetShapeId(...): config vector is empty");

        // enable aov shape id output from OpenCl
        m_cfgs[m_primary].renderer->SetOutput(
            Renderer::OutputType::kShapeId, m_shape_id_data.tmp_output.get());
        m_shape_id_pos = RadeonRays::float2((float)x, (float)y);
        // request shape id from render
        m_shape_id_requested = true;
        return m_promise.get_future();
    }

    Baikal::Shape::Ptr AppClRender::GetShapeById(int shape_id)
    {
        if (shape_id < 0)
            return nullptr;

        // find shape in scene by its id
        for (auto iter = m_scene->CreateShapeIterator(); iter->IsValid(); iter->Next())
        {
            auto shape = iter->ItemAs<Shape>();
            if (shape->GetId() == static_cast<std::size_t>(shape_id))
                return shape;
        }
        return nullptr;
    }

#ifdef ENABLE_DENOISER


    void AppClRender::RestoreDenoiserOutput(std::size_t cfg_index, Renderer::OutputType type) const
    {
//        switch (type)
//        {
//        case Renderer::OutputType::kWorldShadingNormal:
//            m_cfgs[cfg_index].renderer->SetOutput(type, m_outputs[cfg_index].output_normal.get());
//            break;
//        case Renderer::OutputType::kWorldPosition:
//            m_cfgs[cfg_index].renderer->SetOutput(type, m_outputs[cfg_index].output_position.get());
//            break;
//        case Renderer::OutputType::kAlbedo:
//            m_cfgs[cfg_index].renderer->SetOutput(type, m_outputs[cfg_index].output_albedo.get());
//            break;
//        case Renderer::OutputType::kMeshID:
//            m_cfgs[cfg_index].renderer->SetOutput(type, m_outputs[cfg_index].output_mesh_id.get());
//            break;
//        default:
//            // Nothing to restore
//            m_cfgs[cfg_index].renderer->SetOutput(type, nullptr);
//            break;
//        }
    }
#endif
} // Baikal
