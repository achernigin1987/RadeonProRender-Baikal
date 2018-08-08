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
#pragma once

#include "Renderers/renderer.h"
#include "Output/output.h"

#include <map>
#include <set>
#include <string>
#include <stdexcept>
#include <cassert>


namespace Baikal
{
    class Camera;

    /**
    \brief Interface for post-processing effects.

    \details Post processing effects are operating on 2D renderer outputs and 
    do not know anything on rendreing methonds or renderer internal implementation.
    However not all Outputs are compatible with a given post-processing effect, 
    it is done to optimize the implementation and forbid running say OpenCL 
    post-processing effects on Vulkan output. Post processing effects may have 
    scalar parameters and may rely on the presense of certain content type (like 
    normals or diffuse albedo) in a given input set.
    */
    class PostEffect
    {
    public:

        enum class ParamType
        {
            kFloatVal = 0,
            kFloat4Val,
            kStringVal
        };

        struct ParamValue
        {
            float float_value;
            RadeonRays::float4 float4_value;
            std::string str_value;
        };

        class Param
        {
            friend class PostEffect;

        public:
            ParamType GetType() const;

            float GetFloatVal() const;
            RadeonRays::float4 GetFloat4Val() const;
            std::string GetStringVal() const;

        private:
            Param(float value);
            Param(RadeonRays::float4 const& value);
            Param(std::string const& value);

            ParamType m_type;
            ParamValue m_value;
        };

        // Data type to pass all necessary content into the post effect.
        using InputSet = std::map<Renderer::OutputType, Output*>;

        // Specification of the input set types
        using InputTypes = std::set<Renderer::OutputType>;

        // Default constructor & destructor
        PostEffect() = default;
        virtual ~PostEffect() = default;

        virtual InputTypes GetInputTypes() const = 0;

        // Apply post effect and use output for the result
        virtual void Apply(InputSet const& input_set, Output& output) = 0;

        virtual void Update(Camera* camera, unsigned int samples) = 0;

        // Set scalar parameter
        void SetParameter(std::string const& name, float value);
        // Set scalar parameter
        void SetParameter(std::string const& name, const RadeonRays::float4& value);
        // Set string parameter
        void SetParameter(std::string const& name, const std::string& value);

        Param GetParameter(std::string const& name);

    protected:
        // Adds scalar parameter into the parameter map
        void RegisterParameter(std::string const& name, float init_value);
        void RegisterParameter(std::string const& name, RadeonRays::float4 const& init_value);
        // Adds string parameter into the parameter map
        void RegisterParameter(std::string const& name, std::string const& init_value);

    private:

        void SetParameter(std::string const& name, const Param& value);

        void RegisterParameter(std::string const& name, Param const& init_value);
        // Parameter map
        std::map<std::string, Param> m_parameters;
    };
}
