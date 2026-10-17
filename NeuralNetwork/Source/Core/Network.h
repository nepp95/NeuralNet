#pragma once

#include "Core/Layer.h"
#include "Utility/Random.h"

#include <glm/glm.hpp>

#include <span>

namespace NN
{
    inline auto Sigmoid(float sum) -> float
    {
        return 1.0f / (1.0f + (glm::exp(-sum)));
    }

    struct Network
    {
        std::vector<Layer> Layers;

        Network(std::initializer_list<size_t> sizes);

        auto Forward(std::span<const float> input) -> std::span<const float>;
        auto Backward(std::span<const float> target) -> float;
        auto ApplyWeightChanges(std::span<const float> input, float learningRate) -> void;

        auto TrainOnce(std::span<const float> input, std::span<const float> target, float learningRate) -> float;
        auto Train(std::span<const float> input, std::span<const float> target, float learningRate, size_t iterations = 1) -> void;
    };
}
