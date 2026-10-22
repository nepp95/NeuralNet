#pragma once

namespace NN
{
    struct ImageParams
    {
        uint32_t Width = 0;
        uint32_t Height = 0;
        void* Data = nullptr;
    };

    class Image
    {
    public:
        Image(const ImageParams& params);
        Image(const std::filesystem::path& path);
        ~Image();

        auto SetData(void* data) -> void;
        auto GetID() const -> uint32_t { return m_ID; }
        auto GetFramebuffer() const -> uint32_t { return m_Framebuffer; }
        auto GetWidth() const -> uint32_t { return m_Params.Width; }
        auto GetHeight() const -> uint32_t { return m_Params.Height; }

    private:
        ImageParams m_Params;
        uint32_t m_ID = 0;
        uint32_t m_Framebuffer = 0;
    };
}
