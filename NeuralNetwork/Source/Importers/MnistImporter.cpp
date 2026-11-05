#include "pch.h"
#include "MnistImporter.h"

#include <EppoCore/Core/BufferReader.h>
#include <EppoCore/Core/Filesystem.h>
#include <EppoCore/Core/Log.h>

namespace NN
{
    namespace
    {
        constexpr uint32_t ImageMagic = 0x00000803; // unsigned byte, 3 dimensions (count, rows, cols)
        constexpr uint32_t LabelMagic = 0x00000801; // unsigned byte, 1 dimension (count)
        constexpr uint32_t ClassCount = 10;

        struct IdxFile
        {
            std::vector<uint32_t> Dimensions;
            std::vector<uint8_t> Values;
        };

        // idx headers are big-endian
        auto ReadU32(Eppo::BufferReader& reader) -> uint32_t
        {
            std::array<uint8_t, 4> bytes{};
            reader.ReadData(reinterpret_cast<char*>(bytes.data()), bytes.size());
            return (static_cast<uint32_t>(bytes[0]) << 24u) | (static_cast<uint32_t>(bytes[1]) << 16u) |
                (static_cast<uint32_t>(bytes[2]) << 8u) | static_cast<uint32_t>(bytes[3]);
        }

        auto LoadIdx(const std::filesystem::path& path, const uint32_t expectedMagic, const uint32_t dimensionCount)
            -> std::optional<IdxFile>
        {
            Eppo::Buffer buffer = Eppo::ReadBytes(path);
            Eppo::ScopedBuffer guard(buffer);

            if (!buffer)
            {
                Eppo::Log::Error("MnistImporter: failed to read '{}'", path);
                return std::nullopt;
            }

            const uint64_t headerSize = sizeof(uint32_t) * (1 + dimensionCount);
            if (buffer.Size < headerSize)
            {
                Eppo::Log::Error("MnistImporter: '{}' is too small for an idx header", path);
                return std::nullopt;
            }

            Eppo::BufferReader reader(buffer);

            if (const uint32_t magic = ReadU32(reader); magic != expectedMagic)
            {
                Eppo::Log::Error("MnistImporter: '{}' has magic {:#010x}, expected {:#010x}", path, magic, expectedMagic);
                return std::nullopt;
            }

            IdxFile file;
            uint64_t valueCount = 1;

            for (uint32_t i = 0; i < dimensionCount; i++)
            {
                file.Dimensions.push_back(ReadU32(reader));
                valueCount *= file.Dimensions.back();
            }

            if (buffer.Size - headerSize < valueCount)
            {
                Eppo::Log::Error(
                    "MnistImporter: '{}' holds {} bytes of data, header promises {}", path, buffer.Size - headerSize, valueCount
                );
                return std::nullopt;
            }

            file.Values.resize(valueCount);
            reader.ReadData(reinterpret_cast<char*>(file.Values.data()), file.Values.size());

            return file;
        }
    }

    MnistImporter::MnistImporter(std::filesystem::path imagesPath, std::filesystem::path labelsPath)
        : m_ImagesPath(std::move(imagesPath)), m_LabelsPath(std::move(labelsPath))
    {}

    auto MnistImporter::Import() -> std::optional<Dataset>
    {
        ZoneScoped;

        const auto images = LoadIdx(m_ImagesPath, ImageMagic, 3);
        if (!images)
            return std::nullopt;

        const auto labels = LoadIdx(m_LabelsPath, LabelMagic, 1);
        if (!labels)
            return std::nullopt;

        const uint32_t sampleCount = images->Dimensions[0];
        const uint32_t pixelCount = images->Dimensions[1] * images->Dimensions[2];

        if (labels->Dimensions[0] != sampleCount)
        {
            Eppo::Log::Error("MnistImporter: {} images but {} labels", sampleCount, labels->Dimensions[0]);
            return std::nullopt;
        }

        Dataset dataset;
        dataset.SampleCount = sampleCount;
        dataset.InputCount = pixelCount;
        dataset.OutputCount = ClassCount;

        // Pixels to [0, 1]
        dataset.Inputs.resize(images->Values.size());
        for (size_t i = 0; i < images->Values.size(); i++)
            dataset.Inputs[i] = static_cast<float>(images->Values[i]) / 255.0f;

        // Labels to one-hot
        dataset.Targets.assign(static_cast<size_t>(sampleCount) * ClassCount, 0.0f);
        for (uint32_t s = 0; s < sampleCount; s++)
        {
            const uint8_t label = labels->Values[s];
            if (label >= ClassCount)
            {
                Eppo::Log::Error("MnistImporter: sample {} has label {}, expected 0-9", s, label);
                return std::nullopt;
            }

            dataset.Targets[static_cast<size_t>(s) * ClassCount + label] = 1.0f;
        }

        for (uint32_t c = 0; c < ClassCount; c++)
            dataset.ClassNames.push_back(std::to_string(c));

        if (!dataset.IsValid())
        {
            Eppo::Log::Error("MnistImporter: '{}' contains no samples", m_ImagesPath);
            return std::nullopt;
        }

        Eppo::Log::Info("MnistImporter: loaded {} samples of {} pixels", sampleCount, pixelCount);
        return dataset;
    }
}
