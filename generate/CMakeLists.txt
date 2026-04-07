cmake_minimum_required(VERSION 3.18)
project(gpt_generate)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "C:/libtorch")
find_package(Torch REQUIRED)
add_executable(generate main.cpp)
target_link_libraries(generate ${TORCH_LIBRARIES})
# Windows 
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET generate POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS} $<TARGET_FILE_DIR:generate>)
endif()