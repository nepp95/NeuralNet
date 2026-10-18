#include "pch.h"
#include "Utility/BufferReader.h"

namespace NN
{
    BufferReader::BufferReader(const Buffer buffer, const uint64_t offset)
        : m_Buffer(buffer), m_Offset(offset)
    {}

    auto BufferReader::ReadData(char* outData, const size_t size) -> bool
    {
        const bool valid = size <= m_Buffer.Size && m_Offset <= m_Buffer.Size - size;
        if (!valid)
            return false;

        if (size == 0)
            return true;

        std::memcpy(outData, m_Buffer.As<uint8_t>() + m_Offset, size);
        m_Offset += size;

        return true;
    }
}
