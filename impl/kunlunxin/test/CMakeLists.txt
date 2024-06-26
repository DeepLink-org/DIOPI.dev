set(DIOPIRT export_runtime)
set(DIOPIFUNCTIONS export_functions)

#add_compile_options(-fno-elide-constructors)
add_subdirectory(${DIOPI_IMPL_DIR}/third_party/pybind11 build)

include_directories(SYSTEM "${DIOPI_TEST_DIR}/diopi_stub/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/../third_party/pybind11/include")

set(FUNCTION_SAVE_PATH "${DIOPI_TEST_DIR}/diopi_stub/csrc")
set(TEST_GEN_PATH "${DIOPI_TEST_DIR}/diopi_stub/codegen")

# config path for Python.h
find_package(Python3 COMPONENTS Interpreter Development)
if (NOT Python3_FOUND)
  message(FATAL_ERROR "No sutiable python library found.")
endif()
message(STATUS "Python lib: ${Python3_LIBRARIES};${Python3_LIBRARY_DIRS}")
message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")
include_directories(SYSTEM "${Python3_INCLUDE_DIRS}")

set(RUNTIME_SRC
    ${FUNCTION_SAVE_PATH}/litert.cpp
    conform_test.cpp
)

set(EXPORT_SRC
    ${FUNCTION_SAVE_PATH}/export_runtime.cpp
)

message("CXX_LITERT_SRC:" ${CXX_LITERT_SRC})

pybind11_add_module(${DIOPIRT} SHARED ${EXPORT_SRC})
add_library(diopirt SHARED ${RUNTIME_SRC})

target_link_libraries(${DIOPIRT} PRIVATE diopirt)
#target_link_libraries(diopirt ${DEVICEIMPL})
# Since all the symbols are declared weak in diopi_impl,
# depending on the compiler and linker version,
# linking `diopi_impl.so` may fail.
# Here, we use no-as-needed to ensure that diopi_impl.so is linked.
#target_link_libraries(${DIOPIRT} -Wl,--no-as-needed ${DEVICEIMPL} -Wl,--as-needed)
target_link_libraries(diopirt -Wl,--no-as-needed ${DEVICEIMPL} -Wl,--as-needed)

add_custom_target(test_code_gen COMMAND python3 ${TEST_GEN_PATH}/gen.py
    --device=kunlunxin)
add_custom_target(functions_copy ALL
    COMMAND ln -f ${FUNCTION_SAVE_PATH}/export_functions.cpp ${PROJECT_SOURCE_DIR}/test
    DEPENDS test_code_gen)

file(TOUCH export_functions.cpp)
set(FUNCTIONS_SRC
    export_functions.cpp
)

pybind11_add_module(${DIOPIFUNCTIONS} SHARED ${FUNCTIONS_SRC})
target_link_libraries(${DIOPIFUNCTIONS} PRIVATE -Wl,--no-as-needed diopirt -Wl,--as-needed)

add_dependencies(${DIOPIFUNCTIONS} functions_copy)
if(${USE_ADAPTOR} STREQUAL "true")
    add_dependencies(${DIOPIFUNCTIONS} adaptor_code_gen)
endif()

file(MAKE_DIRECTORY ${DIOPI_TEST_DIR}/python)
add_custom_target(python_copy ALL
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIFUNCTIONS}> ${DIOPI_TEST_DIR}/python/diopilib/
    COMMAND ln -f ${LIBRARY_OUTPUT_PATH}/$<TARGET_FILE_NAME:${DIOPIRT}> ${DIOPI_TEST_DIR}/python/diopilib/
    DEPENDS ${DIOPIFUNCTIONS} ${DIOPIRT})
