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

MLDenoiseProvider::MLDenoiseProvider(const Config& config, size_t width, size_t height)
           : m_denoiser(config.context,
                        Baikal::PostEffects::CreateMLDenoiser(MLDenoiserInputs::kColorAlbedoNormal8,
                                                              .1f,
                                                              std::string(),
                                                              width,
                                                              height),
                        MLDenoiserInputs::kColorAlbedoNormal8)
{
    auto add_input = [this, &config, width, height](Renderer::OutputType type)
    {
        auto output = config.factory->CreateOutput(static_cast<std::uint32_t>(width),
                                                   static_cast<std::uint32_t>(height));
        config.renderer->SetOutput(type, output.get());
        m_input_set[type] = output.get();
        m_outputs.push_back(std::move(output));
    };

    add_input(Renderer::OutputType::kColor);
    add_input(Renderer::OutputType::kDepth);
    add_input(Renderer::OutputType::kViewShadingNormal);
    add_input(Renderer::OutputType::kGloss);
    add_input(Renderer::OutputType::kAlbedo);
}

void MLDenoiseProvider::Process(Output* output)
{
    assert(output);
    m_denoiser.Apply(m_input_set, *output);
}

void MLDenoiseProvider::Clear(const Config& config)
{
    for (const auto& output : m_outputs)
    {
        config.renderer->Clear(float3(0, 0, 0), *output);
    }
}