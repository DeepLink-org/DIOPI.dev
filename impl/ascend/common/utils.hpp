/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_UTILS_HPP_
#define IMPL_ASCEND_COMMON_UTILS_HPP_
#include <array>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "../ascend_tensor.hpp"
#include "../env_vars.hpp"
#include "../error.hpp"
#include "float16.hpp"

#define DIOPI_CALL(Expr)                                                                                                  \
    do {                                                                                                                  \
        diopiError_t ret = Expr;                                                                                          \
        if (diopiSuccess != ret) {                                                                                        \
            setLastErrorString("%s: %s at %s:%d\n", ::impl::ascend::getDiopiErrorStr(ret), __func__, __FILE__, __LINE__); \
            printf("%s", ascendGetLastErrorString(false));                                                                \
            return ret;                                                                                                   \
        }                                                                                                                 \
    } while (false);

namespace impl {
namespace ascend {

inline bool isIntegralType(const diopiDtype_t& type) { return type < 8; }

inline bool isIntegralTypeWithBool(const diopiDtype_t& type) { return type < 8 || type == 11; }

inline bool isFloatingType(const diopiDtype_t& type) { return (type <= 10 && type >= 8) || type == 12 || type == 13; }

inline bool isOnDevice(diopiDevice_t dev) { return dev == diopiDevice_t::diopi_device; }

inline const char* deviceType2Str(diopiDevice_t dev) { return dev == diopiDevice_t::diopi_device ? "devce" : "host"; }

template <typename T>
diopiScalar_t constructDiopiScalarT(diopiDtype_t dtype, T val) {
    diopiScalar_t scalar;
    scalar.stype = dtype;
    if (isFloatingType(dtype)) {
        scalar.fval = static_cast<double>(val);
    } else {
        scalar.ival = static_cast<int64_t>(val);
    }
    return scalar;
}

template <typename T>
T getValue(const diopiScalar_t* scalar) {
    ASCEND_CHECK_ABORT(scalar != nullptr, "input should not be nullptr");
    if (isIntegralTypeWithBool(scalar->stype)) {
        return static_cast<T>(scalar->ival);
    } else {
        return static_cast<T>(scalar->fval);
    }
}

/**
 * Take the value in diopiScalar_t as a byte array.
 *
 * @param scalar The input scalar.
 * @param dtype Cast the value to the given dtype. If not specified, the original data type of the scalar will be used.
 * @return A pair of (byte array, number of bytes).
 */
std::pair<std::array<std::byte, sizeof(int64_t)>, int64_t> getScalarBytes(const diopiScalar_t* scalar, std::optional<diopiDtype_t> castToDtype = std::nullopt);

aclDataType getAclDataType(diopiDtype_t type);
const char* diopiDtypeToStr(const diopiDtype_t dtype);

// Those methods can generate new AscendTensor, so context is needed.
diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const diopiSize_t* size, diopiDtype_t dtype, diopiDevice_t device = diopi_device);

diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const std::vector<int64_t>& shape, const std::vector<int64_t>& stride, diopiDtype_t dtype,
                        diopiDevice_t device);

diopiError_t makeTensor(diopiContextHandle_t ctx, AscendTensor& dst, const std::vector<int64_t>& shape, diopiDtype_t dtype);

diopiError_t makeTensorLike(diopiContextHandle_t ctx, AscendTensor& dst, const AscendTensor& src, diopiDtype_t dtype = diopi_dtype_unsupported);

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, AscendTensor& dst, const diopiScalar_t* scalar, diopiDevice_t device = diopi_device);

diopiError_t reshape(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst, const std::vector<int64_t>& shape);

AscendTensor reshape(diopiContextHandle_t ctx, const AscendTensor& src, const std::vector<int64_t>& shape);

diopiError_t contiguous(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst, diopiMemoryFormat_t format = diopiMemoryFormat_t::Contiguous);

diopiError_t castTensor(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst);

diopiError_t castTensor(diopiContextHandle_t ctx, const std::vector<AscendTensor>& src, std::vector<AscendTensor>& dst, diopiDtype_t supportDtype);

diopiTensorHandle_t createTensorIfNullptrOrConstCast(diopiContextHandle_t ctx, diopiConstTensorHandle_t in, diopiSize_t& shape, diopiDtype_t dtype,
                                                     bool isFillingRequired, double value);

/**
 * @brief Convert the data type of an AscendTensor src to the specified supported data type dtype.
 *
 * @param ctx              diopiContextHandle_t context handle for executing operations
 * @param src              Source AscendTensor object for data type conversion
 * @param dtype            Target data type (supported data type)
 *
 * @return diopiError_t    Returns diopiSuccess if the conversion is successful; otherwise, returns other error codes.
 */
diopiError_t castTensor(diopiContextHandle_t ctx, AscendTensor& src, diopiDtype_t supportDtype);

diopiError_t aclAsStrided(diopiContextHandle_t ctx, const AscendTensor& src, AscendTensor& dst);

diopiError_t transTensorTo2D(diopiContextHandle_t ctx, AscendTensor& th);

diopiError_t broadcast(diopiContextHandle_t ctx, AscendTensor& out, const AscendTensor& input, const std::vector<int64_t>& size);

diopiError_t broadcast(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const std::vector<int64_t>& size);

std::vector<int64_t> inferSize(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2);

diopiError_t fillNan(diopiContextHandle_t ctx, AscendTensor& src);

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<AscendTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

void* ascendTensorDeviceToHost(diopiContextHandle_t ctx, AscendTensor at);

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_UTILS_HPP_
