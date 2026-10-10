#include "pch.h"
#include "NeuralNetworks/Perceptron.h"

#include "Random.h"

namespace NN
{
    auto Perceptron::Guess() const -> float
    {
        // Weight and sum inputs
        float data = 0.0f;
        for (size_t i = 0; i < m_Data.size(); i++)
            data += m_Data[i] * m_Weights[i];

        // Activate
        return m_ActivationFn(data);
    }

    auto Perceptron::Train(float desired) -> void
    {
        const float guess = Guess();
        const float error = desired - guess;

        for (size_t i = 0; i < m_Data.size(); i++)
            m_Weights[i] += (error * m_Data[i]) * m_LearningConstant;
    }

    auto Perceptron::SetInputs(std::vector<float>&& inputs) -> void
    {
        m_Data = std::move(inputs);

        if (m_Weights.empty())
        {
            m_Weights.resize(m_Data.size());
            for (size_t i = 0; i < m_Weights.size(); i++)
                m_Weights[i] = Utils::GenerateRandomFloat(-1.0f, 1.0f);
        }
    }
}
