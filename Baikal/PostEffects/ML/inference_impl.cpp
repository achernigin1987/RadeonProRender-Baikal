#include "inference_impl.h"

#include <functional>
#include <numeric>


namespace Baikal
{
    namespace PostEffects
    {
        InferenceImpl::InferenceImpl(size_t width, size_t height)
            : m_width(width), m_height(height)
        {
        }

        Buffer InferenceImpl::GetInputBuffer()
        {
            return AllocBuffer(INPUT_CHANNELS);
        }

        void InferenceImpl::PushInput(Buffer&& buffer)
        {
            assert(buffer.shape()[0] == m_height);
            assert(buffer.shape()[1] == m_width);
            assert(buffer.shape()[2] == INPUT_CHANNELS);
        }

        Buffer InferenceImpl::PopOutput()
        {
            return AllocBuffer(OUTPUT_CHANNELS);
        }

        Buffer InferenceImpl::AllocBuffer(size_t channels)
        {
            size_t size = m_width * m_height * channels;
            return Buffer(Buffer::Data(new Buffer::ValueType[size]),
                          {m_height, m_width, channels});
        }
    }
}
