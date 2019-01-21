#pragma once


namespace Baikal {
namespace PostEffects {

class Exception : public std::exception
{
public:
    const char* what() const noexcept override
    {
        return message_.c_str();
    }

    template<class T>
    Exception&& operator<<(T&& value)
    {
        std::ostringstream stream(message_);
        stream << value;
        message_ = stream.str();
        return std::move(*this);
    }

    Exception&& operator<<(const char* value)
    {
        message_ += value;
        return std::move(*this);
    }

    Exception&& operator<<(const std::string& value)
    {
        message_ += value;
        return std::move(*this);
    }

private:
    std::string message_;
};

} // namespace PostEffects
} // namespace Baikal
