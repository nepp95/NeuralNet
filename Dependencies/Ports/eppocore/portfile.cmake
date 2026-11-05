vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO nepp95/EppoCore
    REF 9aad0165805470e3a166cb58eb76503b696cc2bf
    SHA512 220b7fb458839624c8c5b3fe0e7819477e5f00d89d836afab203941877e33195c06fdf58c5880c766b945dc739a5b665f076d25e0147502cdb32fbf69f63f956
    HEAD_REF master
)

# EppoCore has no license file to install
set(VCPKG_POLICY_SKIP_COPYRIGHT_CHECK enabled)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DEPPO_BUILD_GUI=ON
        -DEPPO_BUILD_HEADLESS=ON
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
