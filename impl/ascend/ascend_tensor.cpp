/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "ascend_tensor.hpp"

#include <array>
#include <cstdint>
#include <mutex>
#include <utility>

#include "common/debug.hpp"

namespace impl {
namespace ascend {

bool AscendTensor::isContiguous(diopiMemoryFormat_t format) const {
    if (!defined()) {
        return true;
    }
    int64_t stride = 1;
    int64_t dim = this->dim();
    const auto& strides = stride_;
    const auto& shape = shape_;

    if (format == diopiMemoryFormat_t::Contiguous) {
        for (int64_t i = dim - 1; i >= 0; i--) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }

    } else if (format == diopiMemoryFormat_t::ChannelsLast) {
        if (strides.size() != 4) return false;
        for (auto& i : {1, 3, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                // shape_d != 1 help dealing with shape like [2, 2048, 1, 1]
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast3d) {
        if (strides.size() != 5) return false;
        for (auto& i : {1, 4, 3, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shape[i];
        }
    } else if (format == diopiMemoryFormat_t::ChannelsLast1d) {
        if (strides.size() != 3) {
            return false;
        }
        for (auto& i : {1, 2, 0}) {
            const auto& shapeD = shape[i];
            if (shapeD != 1) {
                if (strides[i] != stride) {
                    return false;
                }
            }
            stride *= shapeD;
        }
    }
    return true;
}

const void* AscendTensor::data() const {
    const void* p = nullptr;
    diopiGetTensorDataConst(tensor_, &p);
    return p;
}

AscendTensor::ShapeType AscendTensor::getAclMemShape() const {
    AscendTensor::ShapeType baseShapeVec;
    if (this->isContiguous()) {
        if (dim() > 0) {
            baseShapeVec.resize(dim());
            for (int64_t i = 0; i < dim(); i++) {
                baseShapeVec[i] = shape(i);
            }
        } else {
            baseShapeVec.push_back(1);
        }

    } else {
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < dim(); i++) {
            if (stride(i) > maxStride) {
                maxStride = stride(i);
                maxIdx = i;
            }
        }
        if (maxStride > 0) {
            baseShapeVec.push_back(shape(maxIdx) * maxStride);
        } else {
            baseShapeVec.push_back(1);
        }
    }
    return baseShapeVec;
}

int64_t AscendTensor::getAclMemBufferSize() const {
    if (this->isContiguous()) {
        if (dim() > 0) {
            return this->numel() * this->elemsize();
        } else {
            return this->elemsize();
        }
    } else {
        int64_t maxStride = 0, maxIdx = -1;
        for (int64_t i = 0; i < dim(); i++) {
            if (stride(i) > maxStride) {
                maxStride = stride(i);
                maxIdx = i;
            }
        }
        if (maxStride > 0) {
            return shape(maxIdx) * maxStride * this->elemsize();
        } else {
            return this->elemsize();
        }
    }
}

aclFormat inferAclDataFormat(int64_t dim, const int64_t* shape, const int64_t* stride, diopiConstTensorHandle_t tensor) {
    static std::once_flag warningFlag;
    if (dim == 5) {
        std::array<int64_t, 5> thStride{stride[0], stride[1], stride[2], stride[3], stride[4]};
        int st = 1;
        std::array<int64_t, 5> ncdhwStride;
        for (auto k : {4, 3, 2, 1, 0}) {
            ncdhwStride[k] = st;
            if (shape[k] == 0) continue;
            if (shape[k] == -1) st = -1;
            if (st != -1) st *= shape[k];
        }
        if (thStride == ncdhwStride) {
            return ACL_FORMAT_NCDHW;
        }

        st = 1;
        std::array<int64_t, 5> ndhwcStride;
        for (auto k : {1, 4, 3, 2, 0}) {
            ndhwcStride[k] = st;
            if (shape[k] == 0) continue;
            if (shape[k] == -1) st = -1;
            if (st != -1) st *= shape[k];
        }
        if (thStride == ndhwcStride) {
            return ACL_FORMAT_NDHWC;
        }
        std::call_once(
            warningFlag, warning, __FILE__, __LINE__, __FUNCTION__, "Acl only support NCDHW or NDHWC format! but get %s", dumpTensor(tensor).c_str());
    } else if (dim == 4) {
        std::array<int64_t, 4> thStride{stride[0], stride[1], stride[2], stride[3]};
        {
            std::array<int64_t, 4> nchwStride;
            int st = 1;
            for (auto k : {3, 2, 1, 0}) {
                nchwStride[k] = st;
                if (shape[k] == 0) continue;
                if (shape[k] == -1) st = -1;
                if (st != -1) st *= shape[k];
            }
            if (thStride == nchwStride) {
                return ACL_FORMAT_NCHW;
            }
        }
        std::array<int64_t, 4> nhwcStride;
        int st = 1;
        for (auto k : {1, 3, 2, 0}) {
            nhwcStride[k] = st;
            if (shape[k] == 0) continue;
            if (shape[k] == -1) st = -1;
            if (st != -1) st *= shape[k];
        }
        if (thStride == nhwcStride) {
            return ACL_FORMAT_NHWC;
        }
        std::call_once(warningFlag, warning, __FILE__, __LINE__, __FUNCTION__, "Acl only support NCHW or NHWC format! but get %s", dumpTensor(tensor).c_str());
    }
    return ACL_FORMAT_ND;
}

}  // namespace ascend
}  // namespace impl
