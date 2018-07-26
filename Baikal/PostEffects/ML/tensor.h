#pragma once

#include <cstddef>
#include <memory>
#include <tuple>


namespace Baikal
{
    namespace PostEffects
    {
        class Tensor
        {
        public:
            using ValueType = float;
            using Data = std::shared_ptr<ValueType>;

            struct Shape
            {
                std::size_t width;
                std::size_t height;
                std::size_t channels;

                friend bool operator==(const Tensor::Shape& lhs, const Tensor::Shape& rhs)
                {
                    return lhs.width == rhs.width &&
                           lhs.height == rhs.height &&
                           lhs.channels == rhs.channels;
                }
            };

            Tensor(Tensor const&) = delete;
            Tensor& operator=(Tensor const&) = delete;

            Tensor()
                : Tensor({0, 0, 0})
            {
            }

            explicit Tensor(Shape shape)
                : Tensor(nullptr, shape)
            {
            }

            Tensor(Data data, Shape shape)
                : m_data(std::move(data))
                , m_shape(shape)
                , m_size(m_shape.width * m_shape.height * m_shape.channels)
            {
            }

            Tensor(Tensor&&) = default ;
            Tensor& operator=(Tensor&&) = default;

            bool empty() const
            {
                return m_data == nullptr;
            }

            ValueType* data() const
            {
                return m_data.get();
            }

            std::size_t size() const
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
