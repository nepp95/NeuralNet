#pragma once

#include "Utility/Buffer.h"

#include <filesystem>
#include <fstream>

namespace NN
{
    inline auto ReadBytes(const std::filesystem::path& path) -> Buffer
    {
        // Open file stream
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return {};

        // Get file size
        const auto fileSize = std::filesystem::file_size(path);
        if (fileSize == 0)
            return {};

        Buffer buffer(fileSize);
        in.read(buffer.As<char>(), fileSize);

        return buffer;
    }
}
