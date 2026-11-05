#pragma once

namespace NN
{
    // Training-ready data: row-major, Inputs = SampleCount * InputCount, Targets = SampleCount * OutputCount.
    // Classification targets are already one-hot.
    struct Dataset
    {
        uint32_t SampleCount = 0;
        uint32_t InputCount = 0;
        uint32_t OutputCount = 0;
        std::vector<float> Inputs;
        std::vector<float> Targets;
        std::vector<std::string> ClassNames; // optional, empty for regression-style data

        [[nodiscard]] auto IsValid() const -> bool
        {
            return SampleCount > 0 && InputCount > 0 && OutputCount > 0 && Inputs.size() == static_cast<size_t>(SampleCount) * InputCount &&
                Targets.size() == static_cast<size_t>(SampleCount) * OutputCount;
        }
    };

    // Base class for every data source. Importers take their configuration (paths, data) through the constructor.
    class DataImporter
    {
    public:
        virtual ~DataImporter() = default;

        [[nodiscard]] virtual auto Import() -> std::optional<Dataset> = 0;
        [[nodiscard]] virtual auto GetName() const -> std::string_view = 0;
    };
}
