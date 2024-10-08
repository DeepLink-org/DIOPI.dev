# Reuse CUDAComputateArch.cmake
CUDA_DETECT_INSTALLED_GPUS(CURRENT_GPU_LIST)

foreach(CC ${CURRENT_GPU_LIST})
  string(STRIP "${CC}" CC) # remove leading spaces

  # See https://developer.nvidia.com/cuda-gpus#compute for Compute Capability
  if(CC VERSION_GREATER_EQUAL 8.0) # Ampere
    set(ENABLE_TORCH_EXT_FLASH_ATTN ON)
    break()
  endif()
endforeach()

set(FLASH_ATTN_LIB_PATH_2_0 "/mnt/cache/share/platform/dep/DIOPI_pytorch/flash-attention2.4.3_pt2.0")
set(FLASH_ATTN_LIB_PATH_2_1 "/mnt/cache/share/platform/dep/DIOPI_pytorch/flash-attention2.4.3_pt2.1")

if(${Torch_VERSION_MAJOR} EQUAL 2 AND ${Torch_VERSION_MINOR} EQUAL 0)
  set(FLASH_ATTN_LIB_PATH "${FLASH_ATTN_LIB_PATH_2_0}")
elseif(${Torch_VERSION_MAJOR} EQUAL 2 AND ${Torch_VERSION_MINOR} EQUAL 1)
  set(FLASH_ATTN_LIB_PATH "${FLASH_ATTN_LIB_PATH_2_1}")
else()
  message(FATAL_ERROR "No valid torch version for setting FLASH_ATTN_LIB_PATH")
endif()

message(STATUS "FLASH_ATTN_LIB_PATH: ${FLASH_ATTN_LIB_PATH}")

if(ENABLE_TORCH_EXT_FLASH_ATTN)
  # Note: it's really a bad idea to hardcode name and path here.
  find_library(
    TORCH_EXT_FLASH_ATTN
    NAMES
    ENV FLASH_ATTN_LIB_NAME # alternative name of library
    flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so # fallback name of library
    PATHS
    ENV FLASH_ATTN_LIB_DIR # alternative path to search
    ${FLASH_ATTN_LIB_PATH} # fallback path
  )
endif()

if(ENABLE_TORCH_EXT_FLASH_ATTN AND TORCH_EXT_FLASH_ATTN)
  message(STATUS "flash-attention FOUND: ${TORCH_EXT_FLASH_ATTN}")
  add_library(diopi_torch_ext_flash_attn SHARED IMPORTED GLOBAL)
  set_target_properties(diopi_torch_ext_flash_attn PROPERTIES
    IMPORTED_LOCATION ${TORCH_EXT_FLASH_ATTN})
  target_link_options(diopi_torch_ext_flash_attn INTERFACE "LINKER:-no-as-needed")

else()
  if(ENABLE_TORCH_EXT_FLASH_ATTN)
    message(STATUS "flash-attention NOTFOUND")
  else()
    message(STATUS "flash-attention DISABLED")
  endif()

  add_library(diopi_torch_ext_flash_attn INTERFACE)
endif()

target_include_directories(diopi_torch_ext_flash_attn INTERFACE include)
