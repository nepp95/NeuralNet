#include "pch.h"
#include "NeuralNetworkLayer.h"

#include <EppoCore/Core/Entrypoint.h>

class NeuralNetworkApp final : public Eppo::Application
{
public:
    explicit NeuralNetworkApp(Eppo::ApplicationSpecification specification)
        : Application(std::move(specification))
    {
        PushLayer(std::make_shared<NN::NeuralNetworkLayer>());
    }
};

Eppo::Application* Eppo::CreateApplication(Eppo::ApplicationCommandLineArgs args)
{
    Eppo::ApplicationSpecification specification;
    specification.Title = "NeuralNetwork";
    specification.Width = 1280;
    specification.Height = 720;
    specification.CommandLineArgs = args;

    return new NeuralNetworkApp(std::move(specification));
}
