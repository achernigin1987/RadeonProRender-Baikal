#include <cstddef>


namespace ML
{
    class Model
    {
    public:
        using ValueType = float;

        virtual void infer(ValueType const* input,
                           std::size_t width,
                           std::size_t height,
                           std::size_t channels,
                           ValueType* output) const = 0;
    };
}

extern "C"
{
    /**
     * Creates and initializes a model.
     *
     * @param model_path Path to a model.
     * @param gpu_memory_fraction Fraction of GPU memory allowed to use
     *                            by versions with GPU support (0..1].
     *                            Use 0 for default configuration.
     * @param visible_devices Comma-delimited list of GPU devices
     *                        accessible for calculations.
     * @return A new model object.
     */
    ML::Model* LoadModel(char const* model_path,
                         float gpu_memory_fraction,
                         char const* visible_devices);
}
