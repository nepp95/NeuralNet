#include "pch.h"
#include "GPU/StorageBuffer.h"

#include <glad/gl.h>

namespace NN
{
    StorageBuffer::StorageBuffer(const size_t size, const void* initialData)
        : m_Size(size)
    {
        ZoneScoped;

        if (m_Size == 0)
        {
            Log::Error("Can't create a storage buffer with size 0!");
            return;
        }

        Allocate(initialData);
    }

    StorageBuffer::~StorageBuffer()
    {
        glDeleteBuffers(1, &m_ID);
    }

    auto StorageBuffer::Allocate(const void* initialData) -> void
    {
        glCreateBuffers(1, &m_ID);
        glNamedBufferStorage(m_ID, static_cast<GLsizeiptr>(m_Size), initialData, GL_DYNAMIC_STORAGE_BIT);

        if (!initialData)
            glClearNamedBufferData(m_ID, GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);
    }

    auto StorageBuffer::SetData(const void* data, const size_t offset, const size_t size) -> void
    {
        ZoneScoped;

        if (!Valid() || !data)
            return;

        if (offset > m_Size || size > m_Size - offset)
        {
            Log::Error("Storage buffer write out of bounds!");
            return;
        }

        glNamedBufferSubData(m_ID, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), data);
    }

    auto StorageBuffer::Read(void* destination, const size_t offset, const size_t size) const -> void
    {
        ZoneScoped;

        if (!Valid() || !destination)
            return;

        if (offset > m_Size || size > m_Size - offset)
        {
            Log::Error("Storage buffer read out of bounds!");
            return;
        }

        glGetNamedBufferSubData(m_ID, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size), destination);
    }

    auto StorageBuffer::Read(void* destination) const -> void
    {
        Read(destination, 0, m_Size);
    }

    auto StorageBuffer::Resize(const size_t size, const void* initialData) -> void
    {
        ZoneScoped;

        if (size == 0)
        {
            Log::Error("Can't resize a storage buffer to size 0!");
            return;
        }

        glDeleteBuffers(1, &m_ID);

        m_Size = size;
        Allocate(initialData);
    }

    auto StorageBuffer::Bind(const uint32_t binding) const -> void
    {
        if (!Valid())
            return;

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, m_ID);
    }
}
