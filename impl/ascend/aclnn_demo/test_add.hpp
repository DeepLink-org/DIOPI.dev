#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "../../proto/include/diopi/diopirt.h"

#include "aclnnop/aclnn_add.h"

int aclnnAddTest(int32_t deviceId, aclrtContext& context, aclrtStream& stream, diopiTensorHandle_t self, diopiTensorHandle_t other, diopiScalar_t* alpha, diopiTensorHandle_t out);