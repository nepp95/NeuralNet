#pragma once

#include "Importers/DataImporter.h"

namespace NN
{
    class MemoryImporter final : public DataImporter
    {
    public:
        MemoryImporter(uint32_t inputCount, uint32_t outputCount, std::vector<float> inputs, std::vector<float> targets);

        [[nodiscard]] auto Import() -> std::optional<Dataset> override;
        [[nodiscard]] auto GetName() const -> std::string_view override { return "Memory"; }

    private:
        uint32_t m_InputCount = 0;
        uint32_t m_OutputCount = 0;
        std::vector<float> m_Inputs;
        std::vector<float> m_Targets;
    };
}
