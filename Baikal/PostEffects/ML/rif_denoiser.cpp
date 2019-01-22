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
#include "rif_denoiser.h"


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

using MemoryLayout = std::vector<std::pair<Renderer::OutputType, int>>;

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

Handle WrapHandle(void* handle)
{
    return Handle(handle, rifObjectDelete);
}

Handle CreateContext(const CLWContext& clw_context)
{
    rif_context context = nullptr;
    cl_device_id device_id;
    CheckStatus("rifCreateContextFromOpenClContext",
                rifCreateContextFromOpenClContext(RIF_API_VERSION,
                                                  clw_context,
                                                  &device_id,
                                                  clw_context.GetCommandQueue(0),
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
    rif_image_filter image_filter = nullptr;
    CheckStatus("rifContextCreateImageFilter",
                rifContextCreateImageFilter(context.get(),
                                            RIF_IMAGE_FILTER_MIOPEN_DENOISE,
                                            &image_filter));
    return WrapHandle(image_filter);
}

void AttachImageFilter(const Handle& cmd_queue,
                       const Handle& image_filter,
                       const Handle& input_image,
                       const Handle& output_image)
{
    CheckStatus("rifCommandQueueAttachImageFilter",
                rifCommandQueueAttachImageFilter(cmd_queue.get(),
                                                 image_filter.get(),
                                                 input_image.get(),
                                                 output_image.get()));
}

void DetachImageFilter(const Handle& cmd_queue, const Handle& image_filter)
{
    CheckStatus("rifCommandQueueDetachImageFilter",
                rifCommandQueueDetachImageFilter(cmd_queue.get(), image_filter.get()));
}

void SetFilterParameter(const Handle& filter, const char* param_name, const Handle& image)
{
    CheckStatus("rifImageFilterSetParameterImage",
                rifImageFilterSetParameterImage(filter.get(), param_name, image.get()));
}

Handle BufferToImage(const Handle& context,
                     const CLWBuffer<float>& buffer,
                     int image_width,
                     int image_height,
                     int image_channels)
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

void ExecuteCmdQueue(const Handle& context, const Handle& cmd_queue)
{
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
    , m_rif_context(CreateContext(m_context))
    , m_rif_cmd_queue(CreateCmdQueue(m_rif_context))
    , m_rif_image_filter(CreateImageFilter(m_rif_context))
    , m_output{{}, {nullptr, nullptr}}
{
    RegisterParameter("gpu_memory_fraction", .1f);
    RegisterParameter("start_spp", 8u);
    RegisterParameter("visible_devices", std::string());
}

void RIFDenoiser::InitInference()
{
    if (m_inputs.empty() || m_inputs.front().cl.GetElementCount() != m_width * m_height * kInputChannels)
    {
        if (m_output.rif != nullptr) // Re-attach scenario
        {
            DetachImageFilter(m_rif_cmd_queue, m_rif_image_filter);
        }

        m_device_cache = CLWBuffer<RadeonRays::float3>::Create(m_context, CL_MEM_READ_WRITE, m_width * m_height);

        m_inputs.clear();

        for (const auto& input : kInputs)
        {
            m_inputs.push_back(CreateImage(input.second /*image_channels*/));
            SetFilterParameter(m_rif_image_filter, GetFilterParamName(input.first), m_inputs.back().rif);
        }

        m_output = CreateImage(3 /*image_channels*/);

        AttachImageFilter(m_rif_cmd_queue, m_rif_image_filter, m_inputs.front().rif, m_output.rif);
    }
}

unsigned RIFDenoiser::ReadSampleCount(const CLWBuffer<RadeonRays::float3>& buffer)
{
    RadeonRays::float3 first_pixel;
    m_context.ReadBuffer(0 /*idx*/, buffer, &first_pixel /*hostBuffer*/, 1 /*elemCount*/).Wait();
    return static_cast<unsigned>(first_pixel.w);
}

void RIFDenoiser::WriteToInputs(const CLWBuffer<float>& dst_buffer,
                                const CLWBuffer<RadeonRays::float3>& src_buffer,
                                int dst_image_channels,
                                int src_image_channels,
                                int channels_to_copy)
{
    CopyInterleaved(this,
                    dst_buffer,
                    Cast<float>(src_buffer),
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

    auto start_spp = GetParameter("start_spp").GetUint();
    auto input = m_inputs.begin();

    auto clw_inference_output = dynamic_cast<ClwOutput*>(&output);
    if (!clw_inference_output)
    {
        throw std::runtime_error("RIFDenoiser::Apply(...): can not cast output");
    }

    for (const auto& input_desc : kInputs)
    {
        auto type = input_desc.first;
        auto size = input_desc.second;

        auto clw_output = dynamic_cast<ClwOutput*>(input_set.at(type));
        auto device_mem = clw_output->data();

        if (type == OutputType::kColor && ReadSampleCount(device_mem) < start_spp)
        {
            m_context.CopyBuffer(0 /*idx*/,
                                 device_mem /*source*/,
                                 clw_inference_output->data() /*dest*/,
                                 0 /* srcOffset */,
                                 0 /* destOffset */,
                                 clw_output->data().GetElementCount()).Wait();
            return;
        }

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
            m_primitives.Normalize(0 /*deviceIdx*/,
                                   Cast<cl_float3>(device_mem) /*input*/,
                                   Cast<cl_float3>(m_device_cache) /*output*/,
                                   static_cast<int>(device_mem.GetElementCount()));

            WriteToInputs(input->cl /*dst_buffer*/,
                          m_device_cache /*src_buffer*/,
                          size /*dst_image_channels*/,
                          4 /*src_image_channels*/,
                          size /*src_channels_to_copy*/);
            break;

        default:
            break;
        }

        ++input;
    }

    ExecuteCmdQueue(m_rif_context, m_rif_cmd_queue);

//#ifdef ML_DENOISER_IMAGES_DIR
//    {
//        static unsigned input_index = 0;
//        auto data = ReadBuffer(m_context, m_output.cl);
//        SaveImage("output", data.data(), data.size(), input_index++);
//    }
//#endif
//
    // Set w = 1
    m_context.FillBuffer(0 /*idx*/,
                         clw_inference_output->data() /*dest*/,
                         RadeonRays::float3(0, 0, 0, 1),
                         clw_inference_output->data().GetElementCount());

    // Copy x, y, z values
    CopyInterleaved(this,
                    Cast<float>(clw_inference_output->data()) /*dst_buffer*/,
                    m_output.cl /*src_buffer*/,
                    m_width,
                    m_height,
                    4 /*dst_image_channels*/,
                    0 /*dst_channels_offset*/,
                    3 /*src_image_channels*/,
                    0 /*scr_channels_offset*/,
                    3 /*num_channels_to_copy*/);
}

void RIFDenoiser::SetParameter(std::string const& name, Param value)
{
    auto param = GetParameter(name);
    PostEffect::SetParameter(name, value);
    m_is_dirty = true;
}

RIFDenoiser::Image RIFDenoiser::CreateImage(int image_channels)
{
    Image image {
        CLWBuffer<float>::Create(m_context, CL_MEM_READ_WRITE, m_width * m_height * image_channels),
        {{}, {}},
    };

    image.rif = BufferToImage(m_rif_context, image.cl, m_width, m_height, image_channels);

    return image;
}

} // namespace PostEffects
} // namespace Baikal
