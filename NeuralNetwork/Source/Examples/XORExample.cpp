#include "pch.h"
#include "XORExample.h"

namespace NN
{
    XORExample::XORExample()
    {
        m_Samples = {
            Sample{ { 0, 0 }, { 0 } },
            Sample{ { 0, 1 }, { 1 } },
            Sample{ { 1, 0 }, { 1 } },
            Sample{ { 1, 1 }, { 0 } },
        };

        m_TestSamples = m_Samples;

        m_Network = std::make_unique<Network>(Network{ 2, 4, 1 });
    }

    auto XORExample::Run(const uint32_t epochs, const float learningRate) -> float
    {
        float lossSum = 0.0f;

        for (uint32_t i = 0; i < epochs; i++)
        {
            for (const auto& s : m_Samples)
                lossSum += m_Network->TrainOnce(s.Inputs, s.Targets, learningRate);
        }

        RecordRun(epochs, lossSum, m_Samples.back().Inputs);

        return lossSum;
    }
}
