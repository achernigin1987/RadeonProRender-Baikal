#include "inference.h"


namespace Baikal
{
    namespace PostEffects
    {
        class InferenceImpl : public Inference
        {
        public:
            explicit InferenceImpl(size_t width, size_t height);

            Buffer GetInputBuffer() override;
            void PushInput(Buffer&& buffer) override;
            Buffer PopOutput() override;

        private:
            Buffer AllocBuffer(size_t channels);

            size_t m_width;
            size_t m_height;
        };
    }
}
