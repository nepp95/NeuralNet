#pragma once

#include <glm/glm.hpp>

namespace NN
{
    class Shader
    {
    public:
        explicit Shader(const std::filesystem::path& path);
        Shader(const Shader&) = delete;
        Shader(const Shader&&) = delete;
        auto operator=(const Shader&) -> Shader& = delete;
        auto operator=(const Shader&&) -> Shader& = delete;
        ~Shader();

        auto Dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) const -> void;
        auto Dispatch1D(uint32_t elementCount) const -> void;

        auto SetInt(std::string_view name, int32_t value) -> void;
        auto SetUInt(std::string_view name, uint32_t value) -> void;
        auto SetFloat(std::string_view name, float value) -> void;
        auto SetVec2(std::string_view name, const glm::vec2& value) -> void;
        auto SetVec3(std::string_view name, const glm::vec3& value) -> void;
        auto SetVec4(std::string_view name, const glm::vec4& value) -> void;
        auto SetMat3(std::string_view name, const glm::mat3& value) -> void;
        auto SetMat4(std::string_view name, const glm::mat4& value) -> void;

        [[nodiscard]] auto GetID() const -> uint32_t { return m_ID; }
        [[nodiscard]] auto GetWorkGroupSize() const -> const glm::uvec3& { return m_WorkGroupSize; }
        [[nodiscard]] auto Valid() const -> bool { return m_ID != 0; }

    private:
        auto Compile(const std::filesystem::path& path) -> void;
        auto ResolveUniform(std::string_view name) -> int32_t;

    private:
        uint32_t m_ID = 0;
        glm::uvec3 m_WorkGroupSize = glm::uvec3(1, 1, 1);
        std::map<std::string, int32_t, std::less<>> m_UniformLocations;
    };
}
