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

#include <Output/output.h>
#include "ml_denoiser.h"

using namespace Baikal;
using namespace Baikal::PostEffects;

#define OUTPUTS_NUM (5)

MLDenoiseProvider::MLDenoiseProvider(const Config& config, size_t width, size_t height)
           : m_denoiser(config.context,
                        Baikal::PostEffects::CreateMLDenoiser(MLDenoiserInputs::kColorDepthNormalGloss7,
                                                              .1f,
                                                              std::string(),
                                                              width,
                                                              height),
                        MLDenoiserInputs::kColorDepthNormalGloss7)
{
    for (int i = 0; i < OUTPUTS_NUM; i++)
    {
        m_outputs.push_back(config.factory.get()->CreateOutput(static_cast<std::uint32_t>(width), 
                                                               static_cast<std::uint32_t>(height)));
    }

    config.renderer->SetOutput(Renderer::OutputType::kColor, m_outputs[0].get());
    config.renderer->SetOutput(Renderer::OutputType::kDepth, m_outputs[1].get());
    config.renderer->SetOutput(Renderer::OutputType::kViewShadingNormal, m_outputs[2].get());
    config.renderer->SetOutput(Renderer::OutputType::kGloss, m_outputs[3].get());
    config.renderer->SetOutput(Renderer::OutputType::kAlbedo, m_outputs[4].get());
}

void MLDenoiseProvider::Process(Output* output)
{
    assert(output);

    Baikal::PostEffect::InputSet input_set;
    input_set[Baikal::Renderer::OutputType::kColor] = m_outputs[0].get();
    input_set[Baikal::Renderer::OutputType::kDepth] = m_outputs[1].get();
    input_set[Baikal::Renderer::OutputType::kViewShadingNormal] = m_outputs[2].get();
    input_set[Baikal::Renderer::OutputType::kGloss] = m_outputs[3].get();
    input_set[Baikal::Renderer::OutputType::kAlbedo] = m_outputs[4].get();

    m_denoiser.Apply(input_set, *output);
}

void MLDenoiseProvider::Clear(const Config& config)
{
    for (const auto& output : m_outputs)
    {
        config.renderer->Clear(float3(0, 0, 0), *output);
    }
}