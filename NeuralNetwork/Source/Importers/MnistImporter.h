#pragma once

namespace NN::Mnist
{
    // t10k-images.idx3-ubyte - 10k images, use for testing
    // t10k-labels.idx1-ubyte - 10k labels, use for testing
    // train-images.idx3-ubyte - 60k images, use for training
    // train-labels.idx1-ubyte - 60k labels, use for training

    inline auto Byteswap32(uint32_t value, void* ptr) -> void
    {
        value =
            ((value & 0xFF000000u) >> 24u) | ((value & 0x00FF0000u) >> 8u) | ((value & 0x0000FF00u) << 8u) | ((value & 0x000000FFu) << 24u);

        std::memcpy(ptr, &value, sizeof(uint32_t));
    }

    inline auto Byteswap64(uint64_t value, void* ptr) -> void
    {
        value = ((value & 0xFF00000000000000u) >> 56u) | ((value & 0x00FF000000000000u) >> 40u) | ((value & 0x0000FF0000000000u) >> 24u) |
            ((value & 0x000000FF00000000u) >> 8u) | ((value & 0x00000000FF000000u) << 8u) | ((value & 0x0000000000FF0000u) << 24u) |
            ((value & 0x000000000000FF00u) << 40u) | ((value & 0x00000000000000FFu) << 56u);

        std::memcpy(ptr, &value, sizeof(uint64_t));
    }

    inline auto VerifyMagic(uint32_t magic, const bool isLabels) -> bool
    {
        constexpr uint32_t expectedImageMagic = 0x00000803;
        constexpr uint32_t expectedLabelMagic = 0x00000801;

        if (isLabels)
        {
            if (std::memcmp(&magic, &expectedLabelMagic, sizeof(uint32_t)) != 0)
                return false;
        }
        else
        {
            if (std::memcmp(&magic, &expectedImageMagic, sizeof(uint32_t)) != 0)
                return false;
        }

        return true;
    }

    inline auto LoadData(bool trainingData) -> std::vector<float>
    {
        std::filesystem::path path = "Assets/MnistChars";
        if (trainingData)
            path /= "train-images.idx3-ubyte";
        else
            path /= "t10k-images.idx3-ubyte";

        Buffer buffer = ReadBytes(path);
        BufferReader reader(buffer);

        // Read metadata
        uint32_t magic = 0;
        if (!reader.ReadRaw<uint32_t>(magic))
            return {};
        Byteswap32(magic, &magic);
        if (!VerifyMagic(magic, false))
            return {};

        uint32_t count = 0;
        if (!reader.ReadRaw<uint32_t>(count))
            return {};
        Byteswap32(count, &count);

        uint32_t rows = 0;
        if (!reader.ReadRaw<uint32_t>(rows))
            return {};
        Byteswap32(rows, &rows);

        uint32_t cols = 0;
        if (!reader.ReadRaw<uint32_t>(cols))
            return {};
        Byteswap32(cols, &cols);

        // Read pixels
        std::vector<uint8_t> raw(count * cols * rows);
        reader.ReadData(reinterpret_cast<char*>(raw.data()), raw.size());
        buffer.Release();

        std::vector<float> data(count * cols * rows);
        for (size_t i = 0; i < raw.size(); i++)
            data[i] = raw[i] / 255.0f;

        return data;
    }

    inline auto LoadLabels(bool trainingData) -> std::vector<float>
    {
        std::filesystem::path path = "Assets/MnistChars";
        if (trainingData)
            path /= "train-labels.idx1-ubyte";
        else
            path /= "t10k-labels.idx1-ubyte";

        Buffer buffer = ReadBytes(path);
        BufferReader reader(buffer);

        // Read metadata
        uint32_t magic = 0;
        if (!reader.ReadRaw<uint32_t>(magic))
            return {};
        Byteswap32(magic, &magic);
        if (!VerifyMagic(magic, true))
            return {};

        uint32_t count = 0;
        if (!reader.ReadRaw<uint32_t>(count))
            return {};
        Byteswap32(count, &count);

        // Read labels
        std::vector<uint8_t> raw(count);
        reader.ReadData(reinterpret_cast<char*>(raw.data()), raw.size());
        buffer.Release();

        std::vector<float> data(count);
        for (size_t i = 0; i < raw.size(); i++)
            data[i] = static_cast<float>(raw[i]);

        return data;
    }
}
