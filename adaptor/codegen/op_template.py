# Copyright (c) 2023, DeepLink.
from code_template import CodeTemplate

class OpTemplate(object):
    operators_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#include "convert.hpp"
#include "adaptors_enum.hpp"
#include "impl_functions.hpp"

// NOLINTBEGIN

${cast_strategy}

${adaptors}

// NOLINTEND

""")

    adaptor_template = CodeTemplate("""\
extern "C" diopiError_t diopi${op_name}(${attrs}) {
    TimeElapsed adaptorTimeElapsed("${op_name}_adaptor",&(getTimeElapsedRecorder().accumulators[impl::ENUM_${op_name_upper}_ADAPTOR]));
    ${new_input}
    {
        TimeElapsed castInputTimeElapsed("${op_name}_cast_input",&(getTimeElapsedRecorder().accumulators[impl::ENUM_${op_name_upper}_CAST_INPUT]));
        ${cast_input}
    }

    ${cast_output}
    diopiError_t ret;
    {
        TimeElapsed opTimeElapsed("${op_name}",&(getTimeElapsedRecorder().accumulators[impl::ENUM_${op_name_upper}]));
        ret = ::impl::${device}::${call_func};
    }
    return ret;
}

""")

    cast_strategy_template = CodeTemplate("""\
class ${cast_name} {
public:
    static bool getDstDtype(diopiDtype_t srcDtype, diopiDtype_t &targetDtype) {
        bool convert = false;
        switch (srcDtype) {
            ${cases}
            default:
                targetDtype = srcDtype;
        }
        return convert;
    }
};
""")

    impl_declaration_template = CodeTemplate("""\
/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_FUNCTIONS_HPP_
#define IMPL_FUNCTIONS_HPP_

#include <diopi/diopirt.h>

// NOLINTBEGIN
namespace impl {
namespace ${device} {

${impl_declaration}

}  // namespace ${device}

namespace composite {
    
${composite_funcs_decl}

}  // namespace composite
}  // namespace impl

// NOLINTEND
#endif  // IMPL_FUNCTIONS_HPP_

""")
    enum_declaration_template = CodeTemplate("""\

#ifndef ENUM_ADAPTOR_HPP_
#define ENUM_ADAPTOR_HPP_

// NOLINTBEGIN
namespace impl {

enum ENUM_ADAPTORS_TIMERS{
  ENUM_CAST_MEMORYFORMAT,
  ENUM_CASTOUT_CONSTRUCT,
  ENUM_CASTOUT_DECONSTRUCT,
  ${enum_declaration,}
  ENUM_ADAPTORS_TOTAL
};

const char* adaptorEnumToName(unsigned idx);

}  // namespace impl

// NOLINTEND
#endif  // ENUM_ADAPTOR_HPP_
""")
    adaptor_timer_template = CodeTemplate("""\
#include "adaptors_enum.hpp"
#include <array>

// NOLINTBEGIN
namespace impl {
    
    const char* adaptorEnumToName(unsigned idx)
    {   
        static std::array<const char*, ENUM_ADAPTORS_TOTAL> timerNames = { \
        "CAST_MEMORYFORMAT",\
        "CASTOUT_CONSTRUCT",\
        "CASTOUT_DECONSTRUCT",\
        ${enum_names}\
        };
        return timerNames[idx];
    } 
}  // namespace impl
// NOLINTEND
""")
