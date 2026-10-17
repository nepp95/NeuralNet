#pragma once

#include <iostream>

namespace NN
{
    class Log
    {
    public:
        static auto Info(const char* message) -> void { std::cout << message << "\n"; }
    };
}
