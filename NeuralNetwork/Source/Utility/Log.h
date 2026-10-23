#pragma once

#include <iostream>

namespace NN
{
    class Log
    {
    public:
        static auto Info(const char* message) -> void { std::cout << "[Info] " << message << "\n"; }
        static auto Warn(const char* message) -> void { std::cout << "[Warning] " << message << "\n"; }
        static auto Error(const char* message) -> void { std::cerr << "[Error] " << message << "\n"; }
    };
}
