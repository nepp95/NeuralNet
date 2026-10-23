#pragma once

#include "Utility/Buffer.h"

#include <filesystem>
#include <fstream>
#include <string>

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

    inline auto ReadText(const std::filesystem::path& path) -> std::string
    {
        // Open file stream
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return {};

        // Get file size
        const auto fileSize = std::filesystem::file_size(path);
        if (fileSize == 0)
            return {};

        std::string text(fileSize, '\0');
        in.read(text.data(), static_cast<std::streamsize>(fileSize));

        const auto readCount = in.gcount();
        text.resize(readCount > 0 ? static_cast<size_t>(readCount) : 0);

        // Strip a UTF-8 BOM so callers can treat the first line as source
        if (text.starts_with("\xEF\xBB\xBF"))
            text.erase(0, 3);

        return text;
    }
}
