#pragma once

#include <NeuralNetwork.h>

namespace NN
{
    class ExampleBase
    {
    public:
        virtual ~ExampleBase() = default;

        virtual auto OnUIRender() -> void {}

        virtual auto Train() -> void = 0;

        [[nodiscard]] auto GetEpochs() const -> const std::vector<float>& { return m_Epochs; }
        [[nodiscard]] auto GetAccuracy() const -> const std::vector<float>& { return m_Accuracy; }
        [[nodiscard]] auto GetLoss() const -> const std::vector<float>& { return m_Loss; }
        [[nodiscard]] auto GetEpochsPerStep() const -> int { return m_EpochsPerStep; }
        auto SetEpochsPerStep(const int epochs) -> void { m_EpochsPerStep = std::max(epochs, 1); }

    protected:
        std::unique_ptr<Network> m_Network = nullptr;
        Dataset m_Dataset;
        Dataset m_TestDataset;
        std::vector<float> m_Outputs;
        int m_EpochsPerStep = 1;

        std::vector<float> m_Epochs;
        std::vector<float> m_Accuracy;
        std::vector<float> m_Loss;
    };
}
