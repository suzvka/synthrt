project(dsinfer-packcge
    LANGUAGES CXX
    VERSION ${DSINFER_VERSION}
    DESCRIPTION "DiffSinger Package Command Line Tool"
)

file(GLOB_RECURSE _src *.h *.cpp)

find_package(syscmdline CONFIG REQUIRED)
find_package(unofficial-bit7z CONFIG REQUIRED)

dsinfer_add_executable(${PROJECT_NAME}
    SOURCES ${_src}
    LINKS_PRIVATE dsinfer syscmdline::syscmdline unofficial::bit7z::bit7z64
        $<BUILD_INTERFACE:inputparser>
        $<BUILD_INTERFACE:wavfile>
    RC_NAME ${PROJECT_NAME}
    RC_DESCRIPTION ${PROJECT_DESCRIPTION}
)

if(WIN32)
    set(VCPKG_BIN_DIR "${CMAKE_SOURCE_DIR}/vcpkg/installed/x64-windows/bin")
    find_file(CORRECT_7ZIP_DLL
        NAMES 7zip.dll
        PATHS ${VCPKG_BIN_DIR}
        NO_DEFAULT_PATH
    )

    if(CORRECT_7ZIP_DLL)
        message(STATUS "Found correct 7zip.dll at: ${CORRECT_7ZIP_DLL}")
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CORRECT_7ZIP_DLL}"
                "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
            COMMENT "Copying correct 7zip.dll to output directory..."
        )
    else()
        message(FATAL_ERROR "Could not find the required '7zip.dll' in ${VCPKG_BIN_DIR}")
    endif()
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE
    TOOL_DESC="${PROJECT_DESCRIPTION}, Version ${PROJECT_VERSION}"
    TOOL_VERSION="${PROJECT_VERSION}"
)

if(DSINFER_INSTALL AND SYNTHRT_INSTALL_DEPLOY)
    # Install 7zip runtime
    if(WIN32)
        set(_so_ext ".dll")
        set(_so_dest ${CMAKE_INSTALL_BINDIR})
    else()
        if(APPLE)
            set(_so_ext ".dylib")
        else()
            set(_so_ext ".so*")
        endif()

        set(_so_dest ${CMAKE_INSTALL_LIBDIR})
    endif()

    string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type_upper)
    get_filename_component(_searching_path
        "${QMSETUP_APPLOCAL_DEPS_PATHS_${_build_type_upper}}"
        ABSOLUTE BASE_DIR ${CMAKE_SOURCE_DIR}
    )

    file(GLOB _seven_zip_lib "${_searching_path}/*7zip*${_so_ext}")
    install(FILES ${_seven_zip_lib}
        DESTINATION ${_so_dest}
    )
endif()

add_dependencies(${PROJECT_NAME} onnxdriver duration pitch variance acoustic vocoder)