/**********************************************************************
 Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

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


#include "PostEffects/ML/rif_denoiser.h"

#include "Output/clwoutput.h"
#include "PostEffects/ML/exception.h"
#include "PostEffects/ML/operations.h"
#include "Renderers/renderer.h"

#include <RadeonImageFilters.h>
#include <RadeonImageFilters_cl.h>

#ifdef BAIKAL_EMBED_KERNELS
#include "embed_kernels.h"
#include "rif_denoiser.h"


#endif


namespace Baikal {
namespace PostEffects {

using OutputType = Renderer::OutputType;
using Handle = RIFDenoiser::Handle;

namespace {

using MemoryLayout = std::vector<std::pair<Renderer::OutputType, std::size_t>>;

const MemoryLayout kColorDepthNormalGloss7{
    {OutputType::kColor, 3},
    {OutputType::kDepth, 1},
    {OutputType::kViewShadingNormal, 2},
    {OutputType::kGloss, 1},
};

const MemoryLayout kColorAlbedoNormal8{
    {OutputType::kColor, 3},
    {OutputType::kAlbedo, 3},
    {OutputType::kViewShadingNormal, 2},
};

const MemoryLayout kColorAlbedoDepthNormal9{
    {OutputType::kColor, 3},
    {OutputType::kAlbedo, 3},
    {OutputType::kDepth, 1},
    {OutputType::kViewShadingNormal, 2},
};

const auto& kInputs = kColorAlbedoDepthNormal9;

const std::size_t kInputChannels = []
{
    std::size_t channels = 0;
    for (const auto& input_desc : kInputs)
    {
        channels += input_desc.second;
    }
    return channels;
}();

const char* StatusString(rif_int status)
{
#define STATUS(CODE) case CODE: return #CODE;

    switch (status)
    {
    STATUS(RIF_SUCCESS);
    STATUS(RIF_ERROR_COMPUTE_API_NOT_SUPPORTED);
    STATUS(RIF_ERROR_OUT_OF_SYSTEM_MEMORY);
    STATUS(RIF_ERROR_OUT_OF_VIDEO_MEMORY);
    STATUS(RIF_ERROR_INVALID_IMAGE);
    STATUS(RIF_ERROR_UNSUPPORTED_IMAGE_FORMAT);
    STATUS(RIF_ERROR_INVALID_GL_TEXTURE);
    STATUS(RIF_ERROR_INVALID_CL_IMAGE);
    STATUS(RIF_ERROR_INVALID_OBJECT);
    STATUS(RIF_ERROR_INVALID_PARAMETER);
    STATUS(RIF_ERROR_INVALID_TAG);
    STATUS(RIF_ERROR_INVALID_CONTEXT);
    STATUS(RIF_ERROR_INVALID_QUEUE);
    STATUS(RIF_ERROR_INVALID_FILTER);
    STATUS(RIF_ERROR_INVALID_FILTER_ARGUMENT_NAME);
    STATUS(RIF_ERROR_UNIMPLEMENTED);
    STATUS(RIF_ERROR_INVALID_API_VERSION);
    STATUS(RIF_ERROR_INTERNAL_ERROR);
    STATUS(RIF_ERROR_IO_ERROR);
    STATUS(RIF_ERROR_INVALID_PARAMETER_TYPE);
    STATUS(RIF_ERROR_UNSUPPORTED);
    default:
        throw Exception() << "Unknown RIF status: " << status;
    };

#undef STATUS
}

const char* GetFilterParamName(OutputType type)
{
    switch (type)
    {
    case OutputType::kColor: return "colorImg";
    case OutputType::kAlbedo: return "albedoImg";
    case OutputType::kDepth: return "depthImg";
    case OutputType::kViewShadingNormal: return "normalsImg";
    case OutputType::kGloss: return "glossImg";
    default:
        throw Exception() << "Unsupported output type: " << static_cast<int>(type);
    }
}

void CheckStatus(const char* function_name, rif_int status)
{
    if (status != RIF_SUCCESS)
    {
        throw std::runtime_error(std::string(function_name) + " failed: " + StatusString(status));
    }
}

rif_int GetDeviceCount()
{
    rif_int device_count;
    CheckStatus("rifGetDeviceCount",
                rifGetDeviceCount(RIF_BACKEND_API_OPENCL, RIF_PROCESSOR_GPU, &device_count));

    if (device_count <= 0)
    {
        throw std::runtime_error("RIF: no devices found");
    }

    return device_count;
}

rif_int SelectDevice(rif_int device_count, const std::string& visible_devices)
{
    rif_int device_idx = 0;
    std::istringstream stream(visible_devices);
    stream >> device_idx; // Ignore errors
    return std::min(device_idx, device_count - 1);
}

Handle WrapHandle(void* handle)
{
    return Handle(handle, rifObjectDelete);
}

Handle CreateContext(rif_int device_idx)
{
    rif_context context = nullptr;
    CheckStatus("rifCreateContext",
                rifCreateContext(RIF_API_VERSION,
                                 RIF_BACKEND_API_OPENCL,
                                 RIF_PROCESSOR_GPU,
                                 device_idx,
                                 nullptr /*cache_path*/,
                                 &context));
    return WrapHandle(context);
}

Handle CreateCmdQueue(const Handle& context)
{
    rif_command_queue cmd_queue = nullptr;
    CheckStatus("rifContextCreateCommandQueue",
                rifContextCreateCommandQueue(context.get(), &cmd_queue));
    return WrapHandle(cmd_queue);
}

Handle CreateImageFilter(const Handle& context)
{
    rif_image_filter image_filter;
    CheckStatus("rifContextCreateImageFilter",
                rifContextCreateImageFilter(context.get(),
                                            RIF_IMAGE_FILTER_MIOPEN_DENOISE,
                                            &image_filter));
    return WrapHandle(image_filter);
}

void SetFilterParameter(const Handle& filter, const char* param_name, const Handle& image)
{
    CheckStatus("rifImageFilterSetParameterImage",
                rifImageFilterSetParameterImage(filter.get(), param_name, image.get()));
}

Handle BufferToImage(const Handle& context,
                     const CLWBuffer<float>& buffer,
                     std::size_t image_width,
                     std::size_t image_height,
                     std::size_t image_channels)
{
    rif_image_desc image_desc {};
    image_desc.image_width = static_cast<rif_uint>(image_width);
    image_desc.image_height = static_cast<rif_uint>(image_height);
    image_desc.num_components = static_cast<rif_uint>(image_channels);
    image_desc.type = RIF_COMPONENT_TYPE_FLOAT32;

    rif_image image = nullptr;
    CheckStatus("rifContextCreateImageFromOpenClMemory",
                rifContextCreateImageFromOpenClMemory(context.get(),
                                                      &image_desc,
                                                      buffer,
                                                      false /*isImage*/,
                                                      &image));
    return WrapHandle(image);
}

void ExecuteImageFilter(const Handle& context,
                        const Handle& cmd_queue,
                        const Handle& image_filter,
                        const Handle& input_image,
                        const Handle& output_image)
{
    CheckStatus("rifCommandQueueAttachImageFilter",
                rifCommandQueueAttachImageFilter(cmd_queue.get(),
                                                 image_filter.get(),
                                                 input_image.get(),
                                                 output_image.get()));

    CheckStatus("rifContextExecuteCommandQueue",
                rifContextExecuteCommandQueue(context.get(),
                                              cmd_queue.get(),
                                              nullptr /*executionFinishedCallbackFunction*/,
                                              nullptr /*data*/,
                                              nullptr /*time*/));
}

} // namespace


RIFDenoiser::RIFDenoiser(const CLWContext& context, const CLProgramManager *program_manager)
#ifdef BAIKAL_EMBED_KERNELS
    : ClwPostEffect(context, program_manager, "denoise", g_denoise_opencl, g_denoise_opencl_headers)
#else
    : ClwPostEffect(context, program_manager, "../Baikal/Kernels/CL/denoise.cl")
#endif
    , m_context(context)
    , m_primitives(context)
    , m_rif_context(nullptr, nullptr)
    , m_rif_cmd_queue(nullptr, nullptr)
    , m_rif_image_filter(nullptr, nullptr)
    , m_rif_out_image{{}, {nullptr, nullptr}}
{
    RegisterParameter("gpu_memory_fraction", .1f);
    RegisterParameter("start_spp", 8u);
    RegisterParameter("visible_devices", std::string());

    auto device_count = GetDeviceCount();
    auto visible_devices = GetParameter("visible_devices").GetString();
    auto device_idx = SelectDevice(device_count, visible_devices);
    m_rif_context = CreateContext(device_idx);
    m_rif_cmd_queue = CreateCmdQueue(m_rif_context);
    m_rif_image_filter = CreateImageFilter(m_rif_image_filter);
}

void RIFDenoiser::InitInference()
{
    // Realloc cache if needed
    size_t bytes_count = sizeof(float) * m_width * m_height * kInputChannels;

    if (m_host_cache.size() != bytes_count)
    {
        m_host_cache.resize(m_width * m_height);

        m_device_cache = {};
        m_device_cache =
            CLWBuffer<RadeonRays::float3>::Create(m_context, CL_MEM_READ_WRITE, m_width * m_height);

        m_last_denoised_image = {};
        m_last_denoised_image =
            CLWBuffer<RadeonRays::float3>::Create(m_context, CL_MEM_READ_WRITE, m_width * m_height);
        m_has_denoised_image = false;

        m_host_cache.resize(m_width * m_height);

        m_inputs.clear();

        for (const auto& input : kInputs)
        {
            m_inputs.push_back({{}, {nullptr, nullptr}});

            m_inputs.back().cl =
                CLWBuffer<float>::Create(m_context, CL_MEM_READ_WRITE, m_width * m_height * input.second);

            m_inputs.back().rif =
                BufferToImage(m_rif_context, m_inputs.back().cl, m_width, m_height, input.second);

            SetFilterParameter(m_rif_image_filter, GetFilterParamName(input.first), m_inputs.back().rif);
        }

        m_rif_out_image.cl =
            CLWBuffer<float>::Create(m_context, CL_MEM_READ_ONLY, m_width * m_height * 3);

        m_rif_out_image.rif =
            BufferToImage(m_rif_context, m_rif_out_image.cl, m_width, m_height, 3);
    }
}

unsigned RIFDenoiser::ReadSampleCount(CLWBuffer<RadeonRays::float3> buffer)
{
    RadeonRays::float3 first_pixel;
    m_context.ReadBuffer<RadeonRays::float3>(0, buffer, &first_pixel, 1).Wait();
    return static_cast<unsigned>(first_pixel.w);
}

void RIFDenoiser::WriteToInputs(CLWBuffer<float> dst_buffer,
                                CLWBuffer<RadeonRays::float3> src_buffer,
                                int dst_image_channels,
                                int src_image_channels,
                                int channels_to_copy)
{
    CopyInterleaved(this,
                    dst_buffer,
                    src_buffer,
                    m_width,
                    m_height,
                    dst_image_channels,
                    0 /*dst_channels_offset*/,
                    src_image_channels,
                    0 /*src_channels_offset*/,
                    channels_to_copy);
}

PostEffect::InputTypes RIFDenoiser::GetInputTypes() const
{
    InputTypes types;
    for (const auto& input_desc : kInputs)
    {
        types.insert(input_desc.first);
    }
    return types;
}


void RIFDenoiser::Apply(InputSet const& input_set, Output& output)
{
//    auto start_spp = GetParameter("start_spp").GetUint();

    if (m_width != input_set.begin()->second->width() ||
        m_height != input_set.begin()->second->height())
    {
        m_width = input_set.begin()->second->width();
        m_height = input_set.begin()->second->height();
        m_is_dirty = true;
    }

    if (m_is_dirty)
    {
        InitInference();
        m_is_dirty = false;
    }

//    bool too_few_samples = false;
    auto input = m_inputs.begin();

    for (const auto& input_desc : kInputs)
    {
        auto type = input_desc.first;
        auto size = input_desc.second;

        auto clw_output = dynamic_cast<ClwOutput*>(input_set.at(type));
        auto device_mem = clw_output->data();

//        if (type == OutputType::kColor && ReadSampleCount(device_mem) < start_spp)
//        {
//            too_few_samples = true;
//            break;
//        }

        switch (type)
        {
        case OutputType::kColor:
        case OutputType::kViewShadingNormal:
        case OutputType::kGloss:
        case OutputType::kAlbedo:
            DivideBySampleCount(this, m_device_cache /*dst_buffer*/, device_mem /*src_buffer*/);

            WriteToInputs(input->cl /*dst_buffer*/,
                          m_device_cache /*src_buffer*/,
                          size /*dst_image_channels*/,
                          4 /*src_image_channels*/,
                          size /*src_channels_to_copy*/);
            break;

        case OutputType::kDepth:
        {
            m_primitives.Normalize(0 /*deviceIdx*/,
                                   Cast<cl_float3>(device_mem),
                                   Cast<cl_float3>(m_device_cache),
                                   (int)device_mem.GetElementCount());

            WriteToInputs(input->cl /*dst_buffer*/,
                          m_device_cache /*src_buffer*/,
                          size /*dst_image_channels*/,
                          4 /*src_image_channels*/,
                          size /*src_channels_to_copy*/);
            break;
        }

        default:
            break;
        }
    }

#ifdef ML_DENOISER_IMAGES_DIR
    static unsigned input_index = 0;
    SaveImage("input", tensor.data(), tensor.size(), input_index++);
#endif

//    if (too_few_samples)
//    {
//        m_start_seq_num = m_last_seq_num + 1;
//        m_has_denoised_image = false;
//    }
//    else
//    {
//        m_context.ReadBuffer<float>(0,
//                                    *m_inputs,
//                                    tensor.data(),
//                                    m_inputs->GetElementCount()).Wait();
//
//        tensor.tag = ++m_last_seq_num;
//        //m_inference->PushInput(std::move(tensor));
//    }

    auto clw_inference_output = dynamic_cast<ClwOutput*>(&output);

    if (!clw_inference_output)
    {
        throw std::runtime_error("MLDenoiser::Apply(...): can not cast output");
    }

    ExecuteImageFilter(m_rif_context,
                       m_rif_cmd_queue,
                       m_rif_image_filter,
                       m_inputs.front().rif,
                       m_rif_out_image.rif);

    m_context.CopyBuffer<float>(0 /*idx*/,
                                m_rif_out_image.cl,
                                Cast<float>(clw_inference_output->data()),
                                0 /* srcOffset */,
                                0 /* destOffset */,
                                m_rif_out_image.cl.GetElementCount()).Wait();

//    auto inference_res = m_inference->PopOutput();
//
//    if (!inference_res.empty() && inference_res.tag >= m_start_seq_num)
//    {
//#ifdef ML_DENOISER_IMAGES_DIR
//        //SaveImage("output", inference_res.data(), inference_res.size(), inference_res.tag);
//#endif
//        auto dest = m_host_cache.data();
//        auto source = inference_res.data();
//        for (auto i = 0u; i < m_width * m_height; ++i)
//        {
//            dest->x = *source++;
//            dest->y = *source++;
//            dest->z = *source++;
//            dest->w = 1;
//            ++dest;
//        }
//
//        m_context->WriteBuffer<float3>(0 /*idx*/,
//                                       *m_last_denoised_image,
//                                       m_host_cache.data(),
//                                       inference_res.size() / 3);
//        m_has_denoised_image = true;
//
//        m_context->CopyBuffer<float3>(0 /*idx*/,
//                                      *m_last_denoised_image,
//                                      clw_inference_output->data(),
//                                      0 /* srcOffset */,
//                                      0 /* destOffset */,
//                                      m_last_denoised_image->GetElementCount()).Wait();
//    }
//    else if (m_has_denoised_image)
//    {
//        m_context->CopyBuffer<float3>(0 /*idx*/,
//                                      *m_last_denoised_image,
//                                      clw_inference_output->data(),
//                                      0 /* srcOffset */,
//                                      0 /* destOffset */,
//                                      m_last_denoised_image->GetElementCount()).Wait();
//    }
//    else
//    {
//        m_context->CopyBuffer<float3>(0 /*idx*/,
//                                      dynamic_cast<ClwOutput*>(input_set.at(OutputType::kColor))->data(),
//                                      clw_inference_output->data(),
//                                      0 /* srcOffset */,
//                                      0 /* destOffset */,
//                                      m_width * m_height).Wait();
//    }
}

void RIFDenoiser::SetParameter(std::string const& name, Param value)
{
    auto param = GetParameter(name);
    PostEffect::SetParameter(name, value);
    m_is_dirty = true;
}

} // namespace PostEffects
} // namespace Baikal
