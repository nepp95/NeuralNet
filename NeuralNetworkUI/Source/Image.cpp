#include "pch.h"
#include "Image.h"

#include <glad/gl.h>

namespace NN
{
    Image::Image(const ImageParams& params)
        : m_Params(params)
    {
        glCreateTextures(GL_TEXTURE_2D, 1, &m_ID);
        glTextureStorage2D(m_ID, 1, GL_RGBA8, m_Params.Width, m_Params.Height);

        if (m_Params.Data)
            glTextureSubImage2D(m_ID, 0, 0, 0, m_Params.Width, m_Params.Height, GL_RGBA, GL_UNSIGNED_BYTE, m_Params.Data);

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
