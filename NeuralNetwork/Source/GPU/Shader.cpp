#include "pch.h"
#include "GPU/Shader.h"

#include <glm/gtc/type_ptr.hpp>
#include <glad/gl.h>

namespace NN
{
    Shader::Shader(const std::filesystem::path& path)
    {
        Compile(path);
    }

    Shader::~Shader()
    {
        glDeleteProgram(m_ID);
    }

    auto Shader::Compile(const std::filesystem::path& path) -> void
    {
        ZoneScoped;

        const std::string text = ReadText(path);
        std::string source;

        // Inline #include "file" relative to the shader, one level
        for (size_t lineStart = 0; lineStart < text.size();)
        {
            const size_t newline = text.find('\n', lineStart);
            const size_t lineEnd = newline == std::string::npos ? text.size() : newline;
            const std::string_view line(text.data() + lineStart, lineEnd - lineStart);

            constexpr std::string_view directive = "#include \"";
            const auto open = line.find(directive);
            const auto close = open == std::string_view::npos ? std::string_view::npos : line.find('"', open + directive.size());

            if (close == std::string_view::npos)
            {
                source += line;
                source += '\n';
            }
            else
            {
                source += ReadText(path.parent_path() / line.substr(open + directive.size(), close - open - directive.size()));
            }

            lineStart = lineEnd + 1;
        }

        const uint32_t shader = glCreateShader(GL_COMPUTE_SHADER);
        const char* sourcePtr = source.c_str();
        glShaderSource(shader, 1, &sourcePtr, nullptr);
        glCompileShader(shader);

        int32_t status = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status == GL_FALSE)
        {
            int32_t length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

            std::string log(length > 0 ? length : 0, '\0');
            glGetShaderInfoLog(shader, length, nullptr, log.data());

            Log::Error(std::format("Failed to compile {}:\n{}", path.string(), log.c_str()).c_str());
            glDeleteShader(shader);
            return;
        }

        const uint32_t program = glCreateProgram();
        glAttachShader(program, shader);
        glLinkProgram(program);
        glDeleteShader(shader);

        glGetProgramiv(program, GL_LINK_STATUS, &status);
        if (status == GL_FALSE)
        {
            int32_t length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

            std::string log(length > 0 ? length : 0, '\0');
            glGetProgramInfoLog(program, length, nullptr, log.data());

            Log::Error(std::format("Failed to link {}:\n{}", path.string(), log.c_str()).c_str());
            glDeleteProgram(program);
            return;
        }

        m_ID = program;

        int32_t workGroupSize[3] = { 1, 1, 1 };
        glGetProgramiv(m_ID, GL_COMPUTE_WORK_GROUP_SIZE, workGroupSize);
        m_WorkGroupSize = glm::uvec3(
            static_cast<uint32_t>(workGroupSize[0]), static_cast<uint32_t>(workGroupSize[1]), static_cast<uint32_t>(workGroupSize[2])
        );
    }

    auto Shader::Dispatch(const uint32_t groupCountX, const uint32_t groupCountY, const uint32_t groupCountZ) const -> void
    {
        ZoneScoped;

        if (!Valid() || groupCountX == 0 || groupCountY == 0 || groupCountZ == 0)
            return;

        glUseProgram(m_ID);
        glDispatchCompute(groupCountX, groupCountY, groupCountZ);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    auto Shader::Dispatch1D(const uint32_t elementCount) const -> void
    {
        if (elementCount == 0)
            return;

        Dispatch((elementCount - 1) / m_WorkGroupSize.x + 1, 1, 1);
    }

    auto Shader::ResolveUniform(const std::string_view name) -> int32_t
    {
        if (!Valid())
            return -1;

        int32_t location = -1;

        if (const auto it = m_UniformLocations.find(name); it != m_UniformLocations.end())
        {
            location = it->second;
        }
        else
        {
            const std::string key(name);
            location = glGetUniformLocation(m_ID, key.c_str());
            m_UniformLocations.emplace(key, location);
        }

        if (location < 0)
            Log::Warn(std::format("Uniform '{}' not found in compute shader", name).c_str());

        return location;
    }

    auto Shader::SetInt(std::string_view name, const int32_t value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniform1i(m_ID, location, value);
    }

    auto Shader::SetUInt(std::string_view name, const uint32_t value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniform1ui(m_ID, location, value);
    }

    auto Shader::SetFloat(std::string_view name, const float value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniform1f(m_ID, location, value);
    }

    auto Shader::SetVec2(std::string_view name, const glm::vec2& value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniform2fv(m_ID, location, 1, glm::value_ptr(value));
    }

    auto Shader::SetVec3(std::string_view name, const glm::vec3& value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniform3fv(m_ID, location, 1, glm::value_ptr(value));
    }

    auto Shader::SetVec4(std::string_view name, const glm::vec4& value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniform4fv(m_ID, location, 1, glm::value_ptr(value));
    }

    auto Shader::SetMat3(std::string_view name, const glm::mat3& value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniformMatrix3fv(m_ID, location, 1, GL_FALSE, glm::value_ptr(value));
    }

    auto Shader::SetMat4(std::string_view name, const glm::mat4& value) -> void
    {
        if (const int32_t location = ResolveUniform(name); location >= 0)
            glProgramUniformMatrix4fv(m_ID, location, 1, GL_FALSE, glm::value_ptr(value));
    }
}
