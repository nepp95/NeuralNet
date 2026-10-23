#pragma once

namespace NN
{
    struct Layer
    {
        size_t InputCount;
        size_t OutputCount;

        // Row major
        std::vector<float> Weights;
        std::vector<float> Biases;

        std::vector<float> WeightedSums;
        std::vector<float> Activations;
        std::vector<float> Deltas;

        Layer(size_t inputCount, size_t outputCount)
            : InputCount(inputCount), OutputCount(outputCount)
        {
            Weights.resize(InputCount * OutputCount);
            Biases.resize(OutputCount, 0.0f);
            WeightedSums.resize(OutputCount);
            Activations.resize(OutputCount);
            Deltas.resize(OutputCount);
        }
    };
}
