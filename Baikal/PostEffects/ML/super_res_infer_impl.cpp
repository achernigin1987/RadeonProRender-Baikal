/**********************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#include "super_res_infer_impl.h"

namespace Baikal
{
    namespace PostEffects
    {

        using uint32_t = std::uint32_t;

        SuperResInferImpl::SuperResInferImpl(uint32_t width, uint32_t height)
        : m_width(width), m_height(height)
        {
            m_cache.reserve(2 * (3 * width * height));
        }

        Tensor::Shape SuperResInferImpl::GetInputShape() const
        {
            return {m_width, m_height, 3};
        }

        Tensor::Shape SuperResInferImpl::GetOutputShape() const
        {
            return {2 * m_width, 2 * m_height, 3};
        }

        Tensor SuperResInferImpl::GetInputTensor()
        {
            auto deleter = [](Tensor::ValueType *data) {
                delete[] data;
            };

            size_t size = 3 * m_width * m_height;

            return Tensor(Tensor::Data(new Tensor::ValueType[size], deleter),
                          {m_width, m_height, 3});
        }

        void SuperResInferImpl::PushInput(Tensor &&input)
        {
            auto input_shape = input.shape();

            assert(input_shape.width == m_width);
            assert(input_shape.height == m_height);

            // upscale in x dimension
            float *src_row = input.data();
            float *tmp_row = m_cache.data();
            for (uint32_t y = 0; y < m_height; y++) {
                for (uint32_t x = 0; x < m_width; x++) {
                    auto src_x = 3 * x;
                    auto dst_x = 2 * src_x;

                    tmp_row[dst_x] = tmp_row[dst_x + 3] = src_row[src_x];
                    tmp_row[dst_x + 1] = tmp_row[dst_x + 4] = src_row[src_x + 1];
                    tmp_row[dst_x + 2] = tmp_row[dst_x + 5] = src_row[src_x + 2];
                }
                src_row += 3 * m_width;
                tmp_row += 2 * (3 * m_width);
            }

            // upscale in y dimension
            src_row = m_cache.data();
            auto dst_row = m_tensor.data();

            for (uint32_t y = 0; y < m_height; y++) {
                for (uint32_t x = 0; x < 2 * (3 * m_width); x++) {
                    dst_row[x + 2 * (3 * m_width)] = dst_row[x] = src_row[x];
                }
                src_row += 2 * (3 * m_width);
                dst_row += 4 * (3 * m_width);
            }
        }

        Tensor SuperResInferImpl::PopOutput()
        {
            return std::move(m_tensor);
        }
    }
}