#include "pch.h"

#include "Application.h"

auto main() -> int
{
    auto app = new NN::Application();

    if (app->Init() == false || app->Run() == false)
    {
        delete app;
        return 1;
    }

    delete app;
    return 0;
}
