#include "pch.h"
#include "MnistExample.h"

#include "Importers/MnistImporter.h"

namespace NN
{
    MnistExample::MnistExample()
    {
        // Load training data
        auto images = Mnist::LoadData(true);
        auto labels = Mnist::LoadLabels(true);

        constexpr size_t pixelCount = 28 * 28;
        const size_t count = images.size() / pixelCount;

        m_Samples.reserve(count);
        for (size_t i = 0; i < count; i++)
        {
            std::vector<float> target(10, 0.0f);
            target[static_cast<size_t>(labels[i])] = 1.0f;

            m_Samples.emplace_back(
                Sample{ std::vector<float>(images.begin() + i * pixelCount, images.begin() + (i + 1) * pixelCount), std::move(target) }
            );
        }

        // Load testing data
        auto testImages = Mnist::LoadData(false);
        auto testLabels = Mnist::LoadLabels(false);

        const size_t testCount = std::min(testImages.size() / pixelCount, testLabels.size());

        m_TestSamples.reserve(testCount);
        for (size_t i = 0; i < testCount; i++)
        {
            std::vector<float> target(10, 0.0f);
            target[static_cast<size_t>(testLabels[i])] = 1.0f;

            m_TestSamples.emplace_back(
                Sample{ std::vector<float>(testImages.begin() + i * pixelCount, testImages.begin() + (i + 1) * pixelCount),
                        std::move(target) }
            );
        }

        m_TestSampleShape = { 28, 28 };

        // Create neural network
        m_Network = std::make_unique<Network>(Network{ 784, 32, 10 });
    }

    auto MnistExample::Run(const uint32_t epochs, const float learningRate) -> float
    {
        if (!m_Network)
            return std::numeric_limits<float>::infinity();

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
