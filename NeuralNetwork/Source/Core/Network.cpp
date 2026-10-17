#include "pch.h"
#include "Core/Network.h"

namespace NN
{
    Network::Network(std::initializer_list<size_t> sizes)
    {
        auto it = sizes.begin();
        size_t inputCount = *it++;

        for (; it != sizes.end(); ++it)
        {
            const size_t outputCount = *it;
            Layers.emplace_back(inputCount, outputCount);

            auto& layer = Layers.back();
            const float limit = glm::sqrt(6.0f / (inputCount + outputCount));

            for (float& w : layer.Weights)
                w = Utils::GenerateRandomFloat(-limit, limit);

            inputCount = outputCount;
        }
    }

    auto Network::Forward(std::span<const float> input) -> std::span<const float>
    {
        const float* prev = input.data();

        for (Layer& layer : Layers)
        {
            for (size_t o = 0; o < layer.OutputCount; ++o)
            {
                const float* weightRow = layer.Weights.data() + o * layer.InputCount;
                float weightedSum = layer.Biases[o];

                for (size_t i = 0; i < layer.InputCount; ++i)
                    weightedSum += weightRow[i] * prev[i];

                layer.WeightedSums[o] = weightedSum;
                layer.Activations[o] = Sigmoid(weightedSum);
            }

            prev = layer.Activations.data();
        }

        return Layers.back().Activations;
    }

    auto Network::Backward(std::span<const float> target) -> float
    {
        Layer& outputLayer = Layers.back();
        float loss = 0.0f;

        // Output delta = prediction - target
        for (size_t j = 0; j < outputLayer.OutputCount; ++j)
        {
            const float weightedSum = outputLayer.WeightedSums[j];
            const float expectedOutput = target[j];

            loss += glm::max(weightedSum, 0.0f) - weightedSum * expectedOutput + std::log1p(glm::exp(-glm::abs(weightedSum)));
            outputLayer.Deltas[j] = outputLayer.Activations[j] - expectedOutput;
        }

        // Hidden layers
        for (size_t l = Layers.size() - 1; l > 0; --l)
        {
            Layer& currentLayer = Layers[l - 1];
            const Layer& nextLayer = Layers[l];

            std::fill(currentLayer.Deltas.begin(), currentLayer.Deltas.end(), 0.0f);

            for (size_t k = 0; k < nextLayer.OutputCount; ++k)
            {
                const float* weightRow = nextLayer.Weights.data() + k * nextLayer.InputCount;

                for (size_t j = 0; j < currentLayer.OutputCount; ++j)
                    currentLayer.Deltas[j] += weightRow[j] * nextLayer.Deltas[k];
            }

            for (size_t j = 0; j < currentLayer.OutputCount; ++j)
            {
                const float activation = currentLayer.Activations[j];
                currentLayer.Deltas[j] *= activation * (1.0f - activation);
            }
        }

        return loss;
    }

    auto Network::ApplyWeightChanges(std::span<const float> input, float learningRate) -> void
    {
        const float* prev = input.data();

        for (Layer& layer : Layers)
        {
            for (size_t o = 0; o < layer.OutputCount; ++o)
            {
                float* weightRow = layer.Weights.data() + o * layer.InputCount;
                const float change = learningRate * layer.Deltas[o];

                for (size_t i = 0; i < layer.InputCount; ++i)
                    weightRow[i] -= change * prev[i];

                layer.Biases[o] -= change;
            }

            prev = layer.Activations.data();
        }
    }

    auto Network::TrainOnce(std::span<const float> input, std::span<const float> target, float learningRate) -> float
    {
        Forward(input);
        const float loss = Backward(target);
        ApplyWeightChanges(input, learningRate);

        return loss;
    }

    auto Network::Train(std::span<const float> input, std::span<const float> target, float learningRate, size_t iterations) -> void
    {
        for (size_t i = 0; i < iterations; i++)
            TrainOnce(input, target, learningRate);
    }
}
