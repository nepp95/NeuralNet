#pragma once

namespace NN
{
    class StorageBuffer
    {
    public:
        explicit StorageBuffer(size_t size, const void* initialData = nullptr);
        StorageBuffer(const StorageBuffer&) = delete;
        StorageBuffer(const StorageBuffer&&) = delete;
        auto operator=(const StorageBuffer&) -> StorageBuffer& = delete;
        auto operator=(const StorageBuffer&&) -> StorageBuffer& = delete;
        ~StorageBuffer();

        auto SetData(const void* data, size_t offset, size_t size) -> void;
        auto Read(void* destination, size_t offset, size_t size) const -> void;
        auto Read(void* destination) const -> void;

        // Recreates the buffer, so any binding made through Bind is stale
        auto Resize(size_t size, const void* initialData = nullptr) -> void;
        auto Bind(uint32_t binding) const -> void;

        [[nodiscard]] auto GetID() const -> uint32_t { return m_ID; }
        [[nodiscard]] auto GetSize() const -> size_t { return m_Size; }
        [[nodiscard]] auto Valid() const -> bool { return m_ID != 0; }

    private:
        auto Allocate(const void* initialData) -> void;

    private:
        uint32_t m_ID = 0;
        size_t m_Size = 0;
    };
}
