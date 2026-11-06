#pragma once

#include "Examples/ExampleBase.h"

namespace NN
{
    class DatasetExample : public ExampleBase
    {
    public:
        auto Train() -> void override;

        auto OnSettingsRender() -> void override;
        auto OnUIRender() -> void override;

    protected:
        Dataset m_Dataset;
        Dataset m_TestDataset;
        std::vector<float> m_Outputs;
        int m_EpochsPerStep = 1;

    private:
        std::vector<float> m_Epochs;
        std::vector<float> m_Accuracy;
        std::vector<float> m_Loss;
    };
}
