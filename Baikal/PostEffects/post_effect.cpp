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

#include "post_effect.h"

namespace Baikal
{

    PostEffect::Param::Param(float value)
                      : m_type(ParamType::kFloatVal)
    {
        m_value.float_value = value;
    }

    PostEffect::Param::Param(RadeonRays::float4 const& value)
            : m_type(ParamType::kFloat4Val)
    {
        m_value.float4_value = value;
    }

    PostEffect::Param::Param(std::string const& value)
            : m_type(ParamType::kStringVal)
    {
        m_value.str_value = value;
    }

    PostEffect::ParamType PostEffect::Param::GetType() const
    {
        return m_type;
    }

    float PostEffect::Param::GetFloatVal() const
    {
        if (m_type != ParamType::kFloatVal)
            throw std::runtime_error("Attempt to get incorrect param type value");

        return m_value.float_value;
    }

    RadeonRays::float4 PostEffect::Param::GetFloat4Val() const
    {
        if (m_type != ParamType::kFloat4Val)
            throw std::runtime_error("Attempt to get incorrect param type value");

        return m_value.float4_value;
    }

    std::string PostEffect::Param::GetStringVal() const
    {
        if (m_type != ParamType::kStringVal)
            throw std::runtime_error("Attempt to get incorrect param type value");

        return m_value.str_value;
    }

    PostEffect::Param PostEffect::GetParameter(std::string const& name)
    {
        auto iter = m_parameters.find(name);

        if (iter == m_parameters.cend())
        {
            throw std::runtime_error("PostEffect: no such parameter " + name);
        }

        return iter->second;
    }

    void PostEffect::SetParameter(std::string const& name, const Param& value)
    {
        auto iter = m_parameters.find(name);

        if (iter == m_parameters.cend())
        {
            throw std::runtime_error("PostEffect: no such parameter " + name);
        }

        iter->second = value;
    }

    void PostEffect::RegisterParameter(std::string const &name, Param const& initial_value)
    {
        assert(m_parameters.find(name) == m_parameters.cend());

        m_parameters.emplace(name, initial_value);
    }

    void PostEffect::SetParameter(std::string const& name, float value)
    {
        SetParameter(name, Param(value));
    }

    void PostEffect::SetParameter(std::string const& name, RadeonRays::float4 const& value)
    {
        SetParameter(name, Param(value));
    }

    void PostEffect::SetParameter(std::string const& name, std::string const& value)
    {
        SetParameter(name, Param(value));
    }

    void PostEffect::RegisterParameter(std::string const& name, float init_value)
    {
        RegisterParameter(name, Param(init_value));
    }

    void PostEffect::RegisterParameter(std::string const& name, RadeonRays::float4 const& init_value)
    {
        RegisterParameter(name, Param(init_value));
    }

    void PostEffect::RegisterParameter(std::string const& name, std::string const& init_value)
    {
        RegisterParameter(name, Param(init_value));
    }
}