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
#include <iostream>
#include <cassert>
#include <cstring>

namespace Baikal
{
    namespace PostEffects
    {
        static int count = 1;
        using uint32_t = std::uint32_t;
        const uint32_t cnannels_num = 4;

        SuperResInferImpl::SuperResInferImpl(uint32_t width, uint32_t height)
        : m_width(width), m_height(height)
        {
            m_cache.reserve(2 * cnannels_num  * width * height);
        }

        Tensor::Shape SuperResInferImpl::GetInputShape() const
        {
            return {m_width, m_height, cnannels_num };
        }

        Tensor::Shape SuperResInferImpl::GetOutputShape() const
        {
            return {2 * m_width, 2 * m_height, cnannels_num };
        }

        Tensor SuperResInferImpl::GetInputTensor()
        {
            auto deleter = [](Tensor::ValueType *data) {
                delete[] data;
            };

            size_t size = cnannels_num * m_width * m_height;

            return Tensor(Tensor::Data(new Tensor::ValueType[size], deleter),
                          {m_width, m_height, cnannels_num});
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
                    auto src_x = cnannels_num  * x;
                    auto dst_x = 2 * src_x;

                    tmp_row[dst_x] = tmp_row[dst_x + cnannels_num ] = src_row[src_x];
                    tmp_row[dst_x + 1] = tmp_row[dst_x + cnannels_num + 1] = src_row[src_x + 1];
                    tmp_row[dst_x + 2] = tmp_row[dst_x + cnannels_num + 2] = src_row[src_x + 2];
                    tmp_row[dst_x + 3] = tmp_row[dst_x + cnannels_num + 3] = src_row[src_x + 3];
                }
                src_row += cnannels_num * m_width;
                tmp_row += 2 * cnannels_num * m_width;
            }

            // upscale in y dimension
            auto size = 4 * cnannels_num * m_width * m_height;
            m_tensor = Tensor(Tensor::Data(new Tensor::ValueType[size],
                                           [](Tensor::ValueType *data) {
                                               delete[] data;
                                           }),
                                           {m_width, m_height, cnannels_num});

            src_row = m_cache.data();
            auto dst_row = m_tensor.data();

            for (uint32_t y = 0; y < m_height; y++) {
                for (uint32_t x = 0; x < 2 * cnannels_num  * m_width; x++) {
                    dst_row[x + 2 * cnannels_num * m_width]  = dst_row[x] = src_row[x];
                }
                src_row += 2 * cnannels_num * m_width;
                dst_row += 4 * cnannels_num * m_width;
            }

            ++count;
        }

        Tensor SuperResInferImpl::PopOutput()
        {
            return std::move(m_tensor);
        }
    }
}