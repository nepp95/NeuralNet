#include "pch.h"
#include "NeuralNetworks/MultiNeuronExample.h"

#include "Random.h"

#include <glm/glm.hpp>
#include <implot.h>

namespace NN
{
    Neuron::Neuron(uint32_t inputCount)
    {
        assert(inputCount > 0);

        inputCount += 1; // for bias

        Weights.resize(inputCount);
        for (uint32_t i = 0; i < inputCount; i++)
            Weights[i] = Utils::GenerateRandomFloat(-1.0f, 1.0f);
    }

    Layer::Layer(const uint32_t neuronCount, const uint32_t inputCount)
    {
        assert(neuronCount > 0 && inputCount > 0);

        Neurons.reserve(neuronCount);
        for (uint32_t i = 0; i < neuronCount; i++)
            Neurons.emplace_back(inputCount);
    }

    auto Layer::Guess() const -> std::vector<float>
    {
        // Weight and sum inputs for each perceptron
        std::vector<float> output(Neurons.size(), 0.0f);

        for (size_t i = 0; i < Neurons.size(); i++)
        {
            const auto& neuron = Neurons.at(i);

            for (size_t j = 0; j < Data.size(); j++)
                output[i] += Data[j] * neuron.Weights[j];

            output[i] = ActivationFn(output[i]);
        }

        return output;
    }

    MultiNeuronExample::MultiNeuronExample()
    {
        m_Layers.emplace_back(Layer{ 2, 2 });
        m_Layers.emplace_back(Layer{ 4, 2 });
        m_Layers.emplace_back(Layer{ 1, 4 });
    }

    auto MultiNeuronExample::Run() -> void
    {
        // We have a set money budget: 100
        // We have different products with different prices
        // We have a shipping cost: 10-40
        // Is the item within budget?
        if (m_LastMistakes == 0)
            return;

        SetInitialData();
        uint32_t mistakes = ForwardPass();

        m_LastMistakes = mistakes;
        m_Mistakes.emplace_back(static_cast<float>(mistakes));
        m_Epochs++;
    }

    auto MultiNeuronExample::Test(const uint32_t testCount) -> uint32_t
    {
        return 0;
    }

    auto MultiNeuronExample::RenderVisualisation() -> void
    {
        if (m_Inputs.empty() || m_Layers.empty())
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

        float maxPrice = 0.0f;
        float maxShipping = 0.0f;
        ComputeMagnitudes(maxPrice, maxShipping);

        // Evaluate the trained network over the plot area (row 0 = top = max price)
        constexpr int cols = 23;
        constexpr int rows = 70;
        constexpr float shippingMin = 0.0f;
        constexpr float shippingMax = 45.0f;
        constexpr float priceMin = 0.0f;
        constexpr float priceMax = 140.0f;
        const float shippingStep = (shippingMax - shippingMin) / cols;
        const float priceStep = (priceMax - priceMin) / rows;

        std::vector<float> predictions(static_cast<size_t>(rows) * cols);
        for (int r = 0; r < rows; r++)
        {
            const float price = priceMax - (r + 0.5f) * priceStep;
            for (int c = 0; c < cols; c++)
            {
                const float shipping = shippingMin + (c + 0.5f) * shippingStep;
                predictions[r * cols + c] = Forward(price, shipping, maxPrice, maxShipping);
            }
        }

        if (ImPlot::BeginPlot("Decision boundary", ImVec2(-1.0f, -1.0f)))
        {
            ImPlot::SetupAxes("Shipping", "Price");
            ImPlot::SetupAxisLimits(ImAxis_X1, shippingMin, shippingMax, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, priceMin, priceMax, ImPlotCond_Always);

            // Learned surface: -1 = over budget, +1 = within budget
            ImPlot::PushColormap(ImPlotColormap_RdBu);
            ImPlot::PlotHeatmap(
                "Prediction", predictions.data(), rows, cols, -1.0, 1.0, nullptr, ImPlotPoint(shippingMin, priceMin),
                ImPlotPoint(shippingMax, priceMax)
            );
            ImPlot::PopColormap();

            // Sample points
            ImPlot::PlotScatter("Within budget", withinX.data(), withinY.data(), static_cast<int>(withinX.size()));
            ImPlot::PlotScatter("Over budget", overX.data(), overY.data(), static_cast<int>(overX.size()));

            // Reference line: price = 100 - shipping
            constexpr float referenceX[2] = { 5.0f, 40.0f };
            constexpr float referenceY[2] = { 95.0f, 60.0f };
            ImPlot::PlotLine("Reference", referenceX, referenceY, 2);

            ImPlot::EndPlot();
        }
    }

    auto MultiNeuronExample::SetInitialData() -> void
    {
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

        for (auto& layer : m_Layers)
        {
            layer.ActivationFn = [](float sum) -> float
            {
                return glm::tanh(sum);
            };
        }
    }

    auto MultiNeuronExample::ComputeMagnitudes(float& maxPrice, float& maxShipping) const -> void
    {
        for (size_t i = 0; i < m_Inputs.size(); i += 3)
        {
            const auto& price = m_Inputs.at(i);
            const auto& shipping = m_Inputs.at(i + 1);

            if (price > maxPrice)
                maxPrice = price;
            if (shipping > maxShipping)
                maxShipping = shipping;
        }
    }

    auto MultiNeuronExample::Forward(float price, float shipping, float maxPrice, float maxShipping) -> float
    {
        std::vector<float> input = { price / maxPrice, shipping / maxShipping };

        for (auto& layer : m_Layers)
        {
            input.emplace_back(1.0f);
            layer.Data = input;
            input = layer.Guess();
        }

        return input.at(0);
    }

    auto MultiNeuronExample::ForwardPass() -> uint32_t
    {
        float maxPrice = 0.0f;
        float maxShipping = 0.0f;
        ComputeMagnitudes(maxPrice, maxShipping);

        uint32_t mistakes = 0;
        for (size_t i = 0; i < m_Inputs.size() / 3; i++)
        {
            const size_t index = i * 3;
            const auto& price = m_Inputs.at(index);
            const auto& shipping = m_Inputs.at(index + 1);

            // Normalized so every sample is a unit vector
            std::vector<float> input = { price / maxPrice, shipping / maxShipping };
            std::vector<std::vector<float>> outputs;
            outputs.reserve(m_Layers.size());

            for (auto& layer : m_Layers)
            {
                input.emplace_back(1.0f);
                layer.Data = input;
                auto output = layer.Guess();
                outputs.emplace_back(output);
                input = output;
            }

            const float desired = m_Inputs.at(index + 2);
            const float output = outputs.back().at(0);
            const float result = output > 0.0f ? 1.0f : -1.0f;
            if (result != desired)
                mistakes++;

            Backpropagation(desired, outputs);
        }

        return mistakes;
    }

    auto MultiNeuronExample::Backpropagation(const float desired, const std::vector<std::vector<float>>& outputs) -> void
    {
        // Calculate cost
        const float output = outputs.back().at(0);
        const float outputDelta = desired - output;
        const float cost = 0.5f * outputDelta * outputDelta;

        // Collect deltas
        std::vector<std::vector<float>> deltas(outputs.size());
        deltas.back() = { outputDelta * (1.0f - output * output) };

        for (size_t i = outputs.size() - 1; i-- > 0;)
        {
            deltas.at(i).resize(outputs.at(i).size());
            for (size_t j = 0; j < outputs.at(i).size(); j++)
            {
                float sum = 0.0f;
                for (size_t k = 0; k < outputs.at(i + 1).size(); k++)
                    sum += m_Layers.at(i + 1).Neurons.at(k).Weights.at(j) * deltas.at(i + 1).at(k);
                deltas.at(i).at(j) = (1.0f - outputs.at(i).at(j) * outputs.at(i).at(j)) * sum;
            }
        }

        // Weight updates
        for (size_t i = 0; i < m_Layers.size(); i++)
        {
            for (size_t j = 0; j < outputs.at(i).size(); j++)
            {
                auto& weights = m_Layers.at(i).Neurons.at(j).Weights;
                for (size_t k = 0; k < weights.size(); k++)
                    weights.at(k) += m_Layers.at(i).LearningConstant * deltas.at(i).at(j) * m_Layers.at(i).Data.at(k);
            }
        }
    }
}
