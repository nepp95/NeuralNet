#pragma once

#include "NeuralNetworks/ExampleBase.h"

namespace NN
{
    struct Neuron
    {
        std::vector<float> Weights;

        Neuron(uint32_t inputCount);
    };

    struct Layer
    {
        std::vector<float> Data;
        std::vector<Neuron> Neurons;
        std::function<float(float)> ActivationFn;
        float LearningConstant = 0.01f;

        Layer(uint32_t neuronCount, uint32_t inputCount);

        auto Guess() const -> std::vector<float>;
    };

    class MultiNeuronExample : public ExampleBase
    {
    public:
        MultiNeuronExample();
        ~MultiNeuronExample() override = default;

        auto Run() -> void override;
        auto Test(uint32_t testCount) -> uint32_t override;
        auto RenderVisualisation() -> void override;

    private:
        auto SetInitialData() -> void;
        auto ComputeMagnitudes(float& maxPrice, float& maxShipping) const -> void;
        auto Forward(float price, float shipping, float maxPrice, float maxShipping) -> float;
        auto ForwardPass() -> uint32_t;
        auto Backpropagation(float desired, const std::vector<std::vector<float>>& outputs) -> void;

    private:
        std::vector<Layer> m_Layers;
    };
}
