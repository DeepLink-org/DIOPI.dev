<div align=center>
<img src="../img/deepLink_logo.png">
</div>


# impl

 impl 主要用于芯片厂商基于 proto 进行标准算子实现，芯片厂商可通过封装自身计算库或者调用 ``kernel`` 的方式来实现 proto 定义好的标准算子接口以备后续测试调用和训练框架调用。

 其价值体现在以实现统一接口计算库的形式，来对接不同训练框架。无需考虑不同训练框架特性，可更专注于提升每个功能性算子的性能。

其主要功能如下：
 * 实现 proto 函数接口并编译生成计算库以供测试和训练框架调用


## **实现原理**

如果使用diopi_test来测试算子实现的正确性，则需要实现[diopi/proto/include/diopi/diopirt.h](../proto/include/diopi/diopirt.h)中的和厂商相关的runtime接口(通过注册的形式，详见下文)。
编译脚本在[scripts/build_imp.sh](scripts/build_impl.sh)其中编译选项-DTEST表示是否编译diopi_test测试所需要的代码，如果设置为`-DTEST=ON`，则在编译时会包含测试相关的代码，并生成供diopi_test使用的so文件。如果设置为`-DTEST=OFF`，则编译生成的so文件中没diopi_test所需要的so文件，此时可供上层框架调用（比如[DIPU](https://github.com/DeepLink-org/DIPU)）。
#### 实现 diopi_test 所需运行时函数

  [diopi_test/diopi_stub/include/conform_test.h](../diopi_test/include/conform_test.h) 中提供了运行时所需 C-API 函数声明，用户根据函数声明实现运行时所需函数，以便测试套件能够在芯片上管理内存等资源。该实现部分仅供测试时使用。

<!-- #### 要求实现并注册的函数列表如下

  ```
  typedef int32_t (*create_stream_func_t)(diopiStreamHandle_t*);
  //其中diopiStreamHandle_t为void*类型别名;
  typedef int32_t (*destroy_stream_func_t)(diopiStreamHandle_t);

  typedef void* (*malloc_func_t)(uint64_t);
  typedef void (*free_func_t)(void*);

  typedef int32_t (*memcpy_h2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
  typedef int32_t (*memcpy_d2h_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);
  typedef int32_t (*memcpy_d2d_async_func_t)(diopiStreamHandle_t stream, void* dst, const void* src, uint64_t bytes);

  typedef int32_t (*sync_stream_func_t)(diopiStreamHandle_t stream);

  typedef const char* (*get_last_error_string_func_t)();
  ```
#### 实现函数后进行注册

  实现上述 TEST 所需运行时函数后，通过 `diopi_test/csrc/litert.cpp` 提供的注册函数在 `initLibrary` 中进行注册。示例如下:

  ```
  int32_t initLibrary() {
      // others register function...
      diopiRegisterMemcpyD2DAsyncFunc(cuda_memcpy_d2d_async);
      // others register function...
      return diopiSuccess;
  }
  ``` -->

#### 实现 DIOPI 函数接口

  [proto/include/diopi/functions.h](../proto/include/diopi/functions.h) 根据模型训练和框架开发经验定义了一套标准算子的函数，每一个函数完成一个特定的、需要计算设备参与执行的功能。截止目前，从30个常用模型所需算子的角度出发，定义了所需的常见训练算子。该实现部分会由 diopi_test 测试后接入训练框架，用于真实模型训练。在实现的过程中，芯片厂商可根据自身特性来优化算子的性能。

  另外，proto 提供了如张量，标量等基础数据结构，这些基础数据结构也出现在DIOPI标准算子的参数列表中。而其中一些数据接口如张量 *Tensor*，上下文 *Context* 是不透明数据类型 ***Opaque data type***。 因此 [proto/include/diopi/diopirt.h](../proto/include/diopi/diopirt.h) 提供了一套接口用以获取 *Tensor* 的相关信息或者从上下文 *Context* 请求资源。这套接口设计旨在连接训练框架和 DIOPI 算子库， 由训练框架提供给 DIOPI 算子库。而 diopi_test 将以仅为测试服务的原则实现这套接口。

#### 配置 DIOPI 转换逻辑（可选）

  当`impl/${DEVICE}`下存在文件`convert_config.yaml`时，DIOPI自动转换功能（adaptor）将会开启。
  此功能默认会把diopi中的非连续tensor转为连续tensor后，再去调用`impl/${DEVICE}`下的算子。
  对与elementwise类算子，可能不需要转为连续，此时可layout可设置为ND. 如不需要adaptor转为功能，则删除`impl/${DEVICE}`下的`convert_config.yaml`.

  如果某些算子支持的数据类型(dtype)或者数据格式(layout)有限制，例如仅支持某些数据类型或数据格式，可以通过编写配置文件实现调用接口前后的自动转换，从而对数据类型或数据格式进行转换。转换依赖3个DIOPI接口：`diopiDtypeCast`和`diopiCopyInp`、 `diopiContiguous`，因此必须实现这3个接口。需要注意的是，由于这种转换是通过copy来完成的，所以会有一定的性能损耗。
  此功能的代码逻辑在[adaptor](../adaptor)中，并且在diopi函数接口实现时，需要满足一下条件：

  * 所有包含的diopi函数实现的cpp文件需要放在`impl/${DEVICE}/functions/`目录下 <sup>*</sup>
  * 所有diopi函数实现实现需要放在命名空间`impl::${DEVICE}`下 <sup>*</sup>
  * 需要修改CMakeLists.txt，实习支持adaptor的代码生成与编译，可参考[camb的CMakeLists.txt](camb/CMakeLists.txt)

  [*] 其中${DEVICE}为编译adaptor时，指定的厂商名。


  每个厂商的设备(device)有自己对应的转换规则，具体位于`impl/${DEVICE}/convert_config.yaml`文件，配置内容参考：

  ```
  - common_config:
    dtype: (int64)->int32, (float64)->float32
    layout: NCHW

  - diopiAdd:
      dtype: (int64)->int32, (float64)->float32
      tensor_dtype:
          input：(float64)->float32
          other：(float64，int64)->float32
          out: (float64，int64)->float32
      layout: NHWC
  ```

  配置应用可分为三级：
  1. 全局的、特定设备通用的配置(`common_config`)，该配置作用于所有缺省配置的算子，用于对dtype和layout进行转换。
  2. 算子粒度的配置(`diopiAdd`)，该配置会覆盖通用配置，作用于该算子的所有输入和输出参数，其中缺省的部分沿用通用配置。
  3. 参数粒度配置(`tensor_dtype`)：对于算子特殊的参数可以配置参数粒度的转换规则，此时会覆盖算子粒度的配置。

  ##### **配置项说明**

  1. **dtype**

  可在设备通用配置和算子配置中设置支持的`dtype`，通用配置的选项包括：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bool、complex32、complex64和complex128(具体可以参考adaptor/codegen/gen.py中的str_to_diopi_dtype)字典。算子内各参数可配置的类型为该参数理想情况下支持的所有类型，但如果由于设备硬件或软件栈的限制，导致某些数据类型不支持，则需要将某些的不支持类型转换为支持的类型进行计算。
  ```
  # 该算子不支持int64和float64的参数，需要分别转换为int32和float32进行计算，并会在计算完成后转换回原本类型
  dtype: (int64)->int32, (float64)->float32
  ```
  括号中为不支持的类型，`->`指向转换后的类型，括号中可以有多个类型，表示这些类型都会转换至`->`后的类型。

  2. **layout**

  layout可配置的选项包括NCHW、NCL、NCDHW、NLC、NHWC、NDHWC和ND(具体可以参考adaptor/codegen/gen.py中的str_to_diopi_format字典），其中ND表示对tensor不做任何layout相关处理。后续若有其他layout，DIOPI支持后也可配置。配置中两个可同时包含，表示两种类型都支持，默认值即为都支持，对layout没有特殊要求。layout也可以配置算子和参数粒度的，配置形式如下：
  ```
  layout: NCHW，input(NHWC) # 表示除input这个tensor进行NHWC转化外，其余tensor进行NCHW转化。
  ```

  限制：diopi接口的输入参数为执行diopiTensorHandle_t的指针时，表示此参数是希望在diopi实现内部申请空间，故此diopi接口不会做任何adaptor相关逻辑（包含memory_format转化和dtype转化）
