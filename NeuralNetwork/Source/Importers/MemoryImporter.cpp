#include "pch.h"
#include "MemoryImporter.h"

#include <EppoCore/Core/Log.h>

namespace NN
{
    MemoryImporter::MemoryImporter(
        const uint32_t inputCount, const uint32_t outputCount, std::vector<float> inputs, std::vector<float> targets
    )
        : m_InputCount(inputCount), m_OutputCount(outputCount), m_Inputs(std::move(inputs)), m_Targets(std::move(targets))
    {}

    auto MemoryImporter::Import() -> std::optional<Dataset>
    {
        ZoneScoped;

        if (m_InputCount == 0 || m_OutputCount == 0 || m_Inputs.size() % m_InputCount != 0)
        {
            Eppo::Log::Error("MemoryImporter: {} inputs don't divide into samples of {}", m_Inputs.size(), m_InputCount);
            return std::nullopt;
        }

        Dataset dataset;
        dataset.SampleCount = static_cast<uint32_t>(m_Inputs.size() / m_InputCount);
        dataset.InputCount = m_InputCount;
        dataset.OutputCount = m_OutputCount;
        dataset.Inputs = m_Inputs;
        dataset.Targets = m_Targets;

        if (!dataset.IsValid())
        {
            Eppo::Log::Error(
                "MemoryImporter: expected {} targets for {} samples, got {}", static_cast<size_t>(dataset.SampleCount) * m_OutputCount,
                dataset.SampleCount, m_Targets.size()
            );
            return std::nullopt;
        }

        return dataset;
    }
}
