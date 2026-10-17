#pragma once

#include "Core/Layer.h"
#include "Core/Network.h"
#include "Visualisation/NetworkGraph.h"

namespace NN
{
    struct Sample
    {
        std::vector<float> Inputs;
        std::vector<float> Targets;
    };

    class ExampleBase
    {
    public:
        virtual ~ExampleBase() = default;

        virtual auto Run(uint32_t epochs, float learningRate) -> float = 0;
        virtual auto RenderVisualisation() -> void;

        [[nodiscard]] auto GetSampleSize() const -> size_t { return m_Samples.size(); }
        [[nodiscard]] auto GetSamples() const -> const std::vector<Sample>& { return m_Samples; }
        [[nodiscard]] auto GetGraphOptions() -> GraphViewOptions& { return m_GraphOptions; }

    protected:
        auto RecordRun(uint64_t epochs, float lossSum, std::span<const float> lastInputs) -> void;

        GraphViewOptions m_GraphOptions;
        TrainingHistory m_History;
        uint64_t m_TotalEpochs = 0;
        float m_LastLoss = 0.0f;
        std::vector<float> m_LastInputs;

        std::vector<Sample> m_Samples;
        std::unique_ptr<Network> m_Network = nullptr;
    };
}
