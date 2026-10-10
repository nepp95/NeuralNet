#include "pch.h"
#include "NeuralNetworks/SinglePerceptronExample.h"

#include "Random.h"

#include <glm/glm.hpp>
#include <implot.h>

namespace NN
{
    auto SinglePerceptronExample::Run() -> void
    {
        // We have a set money budget: 100
        // We have different products with different prices
        // We have a shipping cost: 10-40
        // Is the item within budget?

        if (m_Inputs.empty())
        {
            constexpr uint32_t inputCount = 2000;
            constexpr uint32_t budget = 100;
            m_Inputs.resize(inputCount * 3); // 0 = price, 1 = shipping, 2 = desired result
            for (size_t i = 0; i < m_Inputs.size(); i += 3)
            {
                m_Inputs[i + 0] = Utils::GenerateRandomFloat(20.0f, 130.0f);
                m_Inputs[i + 1] = Utils::GenerateRandomFloat(5.0f, 40.0f);
                m_Inputs[i + 2] = m_Inputs[i + 0] + m_Inputs[i + 1] <= budget ? 1.0f : -1.0f;
            }
        }

        if (m_LastMistakes == 0)
            return;

        m_Perceptron.SetActivationFn(
            [](float sum) -> float
            {
                return sum >= 0.0f ? 1.0f : -1.0f;
            }
        );

        uint32_t mistakes = 0;
        for (size_t i = 0; i < m_Inputs.size() / 3; i++)
        {
            const size_t index = i * 3;
            const auto& price = m_Inputs.at(index);
            const auto& shipping = m_Inputs.at(index + 1);
            // Normalized so every sample is a unit vector
            const glm::vec3 input = glm::normalize(glm::vec3(price, shipping, 1.0f));
            m_Perceptron.SetInputs({ input.x, input.y, input.z });

            const float desired = m_Inputs.at(index + 2);
            if (m_Perceptron.Guess() != desired)
                mistakes++;
            m_Perceptron.Train(desired);
        }

        m_LastMistakes = mistakes;
        m_Mistakes.emplace_back(static_cast<float>(mistakes));

        m_Epochs++;
    }

    auto SinglePerceptronExample::Test(const uint32_t testCount) -> uint32_t
    {
        constexpr uint32_t budget = 100;

        uint32_t correct = 0;
        for (uint32_t i = 0; i < testCount; i++)
        {
            const float price = Utils::GenerateRandomFloat(20.0f, 130.0f);
            const float shipping = Utils::GenerateRandomFloat(5.0f, 40.0f);
            const float desired = price + shipping <= budget ? 1.0f : -1.0f;

            const glm::vec3 input = glm::normalize(glm::vec3(price, shipping, 1.0f));
            m_Perceptron.SetInputs({ input.x, input.y, input.z });

            if (m_Perceptron.Guess() == desired)
                correct++;
        }

        return correct;
    }

    auto SinglePerceptronExample::RenderVisualisation() -> void
    {
        if (m_Perceptron.GetWeights().empty())
            return;

        // Split sample points by desired label (x = shipping, y = price)
        std::vector<float> withinX, withinY, overX, overY;
        for (size_t i = 0; i < m_Inputs.size(); i += 3)
        {
            if (m_Inputs[i + 2] > 0.0f)
            {
                withinX.push_back(m_Inputs[i + 1]);
                withinY.push_back(m_Inputs[i]);
            }
            else
            {
                overX.push_back(m_Inputs[i + 1]);
                overY.push_back(m_Inputs[i]);
            }
        }

        // Learned boundary: w0*price + w1*shipping + w2 = 0  ->  price = -(w1*shipping + w2) / w0
        const auto& weights = m_Perceptron.GetWeights();
        const float wPrice = weights[0];
        const float wShipping = weights[1];
        const float wBias = weights[2];

        if (ImPlot::BeginPlot("Decision boundary", ImVec2(-1.0f, -1.0f)))
        {
            ImPlot::SetupAxes("Shipping", "Price");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 45.0, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 140.0, ImPlotCond_Always);

            // Sample points
            ImPlot::PlotScatter("Within budget", withinX.data(), withinY.data(), static_cast<int>(withinX.size()));
            ImPlot::PlotScatter("Over budget", overX.data(), overY.data(), static_cast<int>(overX.size()));

            // Reference line: price = 100 - shipping
            constexpr float referenceX[2] = { 5.0f, 40.0f };
            constexpr float referenceY[2] = { 95.0f, 60.0f };
            ImPlot::PlotLine("Reference", referenceX, referenceY, 2);

            // Learned line at shipping = 5 and 40
            const float learnedX[2] = { 5.0f, 40.0f };
            const float learnedY[2] = { -(wShipping * 5.0f + wBias) / wPrice, -(wShipping * 40.0f + wBias) / wPrice };
            ImPlot::PlotLine("Learned", learnedX, learnedY, 2);

            ImPlot::EndPlot();
        }
    }
}
