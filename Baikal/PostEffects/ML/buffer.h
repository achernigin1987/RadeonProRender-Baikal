#pragma once

#include <vector>
#include <cstddef>
#include <memory>


namespace Baikal
{
    namespace PostEffects
    {
        class Buffer
        {
        public:
            using ValueType = float;
            using Data = std::unique_ptr<ValueType[]>;
            using Shape = std::tupel<size_t, size_t, size_t>;

            Buffer(Data data, Shape shape) :
                m_data(std::move(data)),
                m_shape(std::move(shape)),
                m_size(std::get<0>(m_shape) *
                       std::get<1>(m_shape) *
                       std::get<2>(m_shape))
            {
            }

            bool empty() const
            {
                return m_size == 0;
            }

            ValueType* data() const
            {
                return m_data.get();
            }

            size_t size() const
            {
                return m_size;
            }

            const Shape& shape() const
            {
                return m_shape;
            }

        private:
            Data m_data;
            Shape m_shape;
            std::size_t m_size;
        };
    }
}
