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
#include "Application/app_utils.h"

#include <iostream>

namespace
{
    char const* kHelpMessage =
        "Baikal [-p path_to_models][-f model_name][-b][-r][-ns number_of_shadow_rays][-ao ao_radius][-w window_width][-h window_height][-nb number_of_indirect_bounces]";
}

namespace Baikal
{
    AppCliParser::AppCliParser(int argc, char * argv[])
    : m_cmd_parser(argc, argv)
    {   }

    AppSettings AppCliParser::Parse()
    {
        AppSettings s;

        s.help = m_cmd_parser.OptionExists("-help");

        s.path = m_cmd_parser.GetOption("-p", s.path);

        s.modelname = m_cmd_parser.GetOption("-f", s.modelname);

        s.envmapname = m_cmd_parser.GetOption("-e", s.envmapname);

        s.width = m_cmd_parser.GetOption("-w", s.width);

        s.height = m_cmd_parser.GetOption("-h", s.height);

        if (m_cmd_parser.OptionExists("-ao"))
        {
            s.ao_radius = m_cmd_parser.GetOption<float>("-ao");
            s.num_ao_rays = (int)(s.ao_radius);
            s.ao_enabled = true;
        }

        s.num_bounces = m_cmd_parser.GetOption("-nb", s.num_bounces);

        s.split_output = m_cmd_parser.OptionExists("-split");

        s.camera_pos.x = m_cmd_parser.GetOption("-cpx", s.camera_pos.x);

        s.camera_pos.y = m_cmd_parser.GetOption("-cpy", s.camera_pos.y);

        s.camera_pos.z = m_cmd_parser.GetOption("-cpz", s.camera_pos.z);

        s.camera_at.x = m_cmd_parser.GetOption("-tpx", s.camera_at.x);

        s.camera_at.y = m_cmd_parser.GetOption("-tpy", s.camera_at.y);

        s.camera_at.z = m_cmd_parser.GetOption("-tpz", s.camera_at.z);

        s.envmapmul = m_cmd_parser.GetOption("-em", s.envmapmul);

        s.num_samples = m_cmd_parser.GetOption("-ns", s.num_samples);

        s.camera_aperture = m_cmd_parser.GetOption("-a", s.camera_aperture);

        s.camera_focus_distance = m_cmd_parser.GetOption("-fd", s.camera_focus_distance);

        s.camera_focal_length = m_cmd_parser.GetOption("-fl", s.camera_focal_length);

        s.camera_out_folder = m_cmd_parser.GetOption("-output_cam", s.camera_out_folder);

        s.camera_sensor_size.x = m_cmd_parser.GetOption("-ssx", s.camera_sensor_size.x);

        s.camera_sensor_size.y = m_cmd_parser.GetOption("-ssy", s.camera_sensor_size.y);

        s.base_image_file_name = m_cmd_parser.GetOption("-ifn", s.base_image_file_name);

        s.image_file_format = m_cmd_parser.GetOption("-iff", s.image_file_format);

        s.gpu_mem_fraction = m_cmd_parser.GetOption("-gmf", s.gpu_mem_fraction);

        s.visible_devices = m_cmd_parser.GetOption("-vds", s.visible_devices);

        auto has_primary_device = [](std::string const& str)
        {
            if (str.empty())
            {
                return true;
            }

            std::stringstream ss(str);

            while (ss.good())
            {
                std::string substr;
                std::getline( ss, substr, ',' );

                if (substr == "0")
                {
                    return true;
                }
            }
            return false;
        };

        float max_gmf = 1.f;
        if (has_primary_device(s.visible_devices))
        {
            max_gmf = .5f;
        }

        if (s.gpu_mem_fraction < 0.0f)
        {
            std::cout << "WARNING: '-gmf' option value clamped to zero" << std::endl;
            s.gpu_mem_fraction = 0.0f;
        }
        else if (s.gpu_mem_fraction > max_gmf)
        {
            std::cout << "WARNING: '-gmf' option value clamped to one or 0.5 in case primary device" << std::endl;
            s.gpu_mem_fraction = 1.f;
        }

        if (m_cmd_parser.OptionExists("-ct"))
        {
            auto camera_type = m_cmd_parser.GetOption("-ct");

            if (camera_type == "perspective")
                s.camera_type = CameraType::kPerspective;
            else if (camera_type == "orthographic")
                s.camera_type = CameraType::kOrthographic;
            else
                throw std::runtime_error("Unsupported camera type");
        }

        s.interop = m_cmd_parser.GetOption<bool>("-interop", s.interop);

        s.cspeed = m_cmd_parser.GetOption("-cs", s.cspeed);

        if (m_cmd_parser.OptionExists("-config"))
        {
            auto cfg = m_cmd_parser.GetOption("-config");

            if (cfg == "cpu")
                s.mode = Mode::kUseSingleCpu;
            else if (cfg == "gpu")
                s.mode = Mode::kUseSingleGpu;
            else if (cfg == "mcpu")
                s.mode = Mode::kUseCpus;
            else if (cfg == "mgpu")
                s.mode = Mode::kUseGpus;
            else if (cfg == "all")
                s.mode = Mode::kUseAll;
        }

        s.platform_index = m_cmd_parser.GetOption("-platform", s.platform_index);

        s.device_index = m_cmd_parser.GetOption("-device", s.device_index);

        if ((s.device_index >= 0) && (s.platform_index < 0))
        {
            std::cout <<
                "Can not set device index, because platform index was not specified" << std::endl;
        }

        if (m_cmd_parser.OptionExists("-r"))
        {
            s.progressive = true;
        }

        if (m_cmd_parser.OptionExists("-nowindow"))
        {
            s.cmd_line_mode = true;
        }

        if (m_cmd_parser.OptionExists("-denoiser_type"))
        {
            auto denoiser_type = m_cmd_parser.GetOption("-denoiser_type");

            if (denoiser_type == "bilateral")
            {
                s.denoiser_type = DenoiserType::kBilateral;
            }
            else if (denoiser_type == "wavelet")
            {
                s.denoiser_type = DenoiserType::kWavelet;
            }
            else if (denoiser_type == "ml")
            {
                s.denoiser_type = DenoiserType::kML;
            }
            else
            {
                std::cerr << "WARNING: unknown denoiser mode\n";
            }
        }

        s.denoiser_start_spp = m_cmd_parser.GetOption("-start_spp", s.denoiser_start_spp);

        return s;
    }

    void AppCliParser::ShowHelp()
    {
        std::cout << kHelpMessage << "\n";
    }

    AppSettings::AppSettings()
        : help(false)
        , path("../Resources/CornellBox")
        , modelname("orig.objm")
        , envmapname("../Resources/Textures/pano_port_001.jpg")
        //render
        , width(800)
        , height(600)
        , num_bounces(9)
        , num_samples(-1)
        , interop(true)
        , cspeed(20.f)
        , mode(Mode::kUseSingleGpu)
        //ao
        , ao_radius(1.f)
        , num_ao_rays(1)
        , ao_enabled(false)

        //camera
        , camera_pos(0.f, 1.f, 3.f)
        , camera_at(0.f, 1.f, 0.f)
        , camera_up(0.f, 1.f, 0.f)
        , camera_sensor_size(0.036f, 0.024f)  // default full frame sensor 36x24 mm
        , camera_zcap(0.0f, 100000.f)
        , camera_aperture(0.f)
        , camera_focus_distance(1.f)
        , camera_focal_length(0.035f) // 35mm lens
        , camera_type (CameraType::kPerspective)
        , camera_out_folder(".")

        //app
        , progressive(false)
        , cmd_line_mode(false)
        , recording_enabled(false)
        , benchmark(false)
        , gui_visible(true)
        , time_benchmarked(false)
        , rt_benchmarked(false)
        , time_benchmark(false)
        , time_benchmark_time(0.f)

        //imagefile
        , base_image_file_name("out")
        , image_file_format("png")

        //unused
        , num_shadow_rays(1)
        , samplecount(0)
        , envmapmul(2.f)
        , platform_index(-1)
        , device_index(-1)
    {
    }
}
