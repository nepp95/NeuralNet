#include "pch.h"
#include "GPU/Image.h"

#include <glad/gl.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace NN
{
    Image::Image(const ImageParams& params)
        : m_Params(params)
    {
        if (m_Params.Width == 0 || m_Params.Height == 0)
        {
            Log::Info("Can't create image with size 0x0!");
            return;
        }

        glCreateTextures(GL_TEXTURE_2D, 1, &m_ID);
        glTextureStorage2D(m_ID, 1, GL_RGBA8, m_Params.Width, m_Params.Height);
        glTextureParameteri(m_ID, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(m_ID, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(m_ID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(m_ID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        if (m_Params.Data)
            glTextureSubImage2D(m_ID, 0, 0, 0, m_Params.Width, m_Params.Height, GL_RGBA, GL_UNSIGNED_BYTE, m_Params.Data);

        glCreateFramebuffers(1, &m_Framebuffer);
        glNamedFramebufferTexture(m_Framebuffer, GL_COLOR_ATTACHMENT0, m_ID, 0);
    }

    Image::Image(const std::filesystem::path& path)
    {
        int width = 0;
        int height = 0;
        auto* data = stbi_load(path.string().c_str(), &width, &height, nullptr, STBI_rgb_alpha);
        if (!data)
        {
            Log::Info((std::string("Failed to load image ") + path.string().c_str() + ": " + stbi_failure_reason()).c_str());
            return;
        }

        m_Params.Width = static_cast<uint32_t>(width);
        m_Params.Height = static_cast<uint32_t>(height);

        glCreateTextures(GL_TEXTURE_2D, 1, &m_ID);
        glTextureStorage2D(m_ID, 1, GL_RGBA8, width, height);
        glTextureSubImage2D(m_ID, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glTextureParameteri(m_ID, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(m_ID, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(m_ID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(m_ID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        stbi_image_free(data);

        glCreateFramebuffers(1, &m_Framebuffer);
        glNamedFramebufferTexture(m_Framebuffer, GL_COLOR_ATTACHMENT0, m_ID, 0);
    }

    Image::~Image()
    {
        glDeleteTextures(1, &m_ID);
        glDeleteFramebuffers(1, &m_Framebuffer);
    }

    auto Image::SetData(void* data) -> void
    {
        glTextureSubImage2D(m_ID, 0, 0, 0, m_Params.Width, m_Params.Height, GL_RGBA, GL_UNSIGNED_BYTE, data);
    }
}
