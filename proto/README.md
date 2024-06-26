<div align=center>
<img src="../img/deepLink_logo.png">
</div>

# proto

 DIOPI中的proto是标准算子接口的原型声明，是芯片厂商实现与框架算子调用的中间层。通过规定标准的运行时函数与算子接口的声明，对框架来说，统一了算子接口，无需考虑芯片厂商具体的算子实现；对厂商来说，可以只聚焦于算子实现与优化，无需考虑框架适配。proto组件作为DIOPI中具体算子声明的环节，起到了承上（框架）启下（芯片厂商）的作用。

 proto组件有如下核心功能：
 1. **实现Runtime标准接口定义**。
 声明了在实现标准算子函数时可以使用的工具函数以及相关数据结构。其中，工具函数用于对Context和Tensor两类对象进行操作。
 2. **实现标准算子的接口定义**。
 声明了标准算子的函数，每一个函数完成一个特定的、需要计算设备参与执行的功能。


proto的主要组成部分包括 _运行时函数(diopirt)_ 和 _算子声明(functions)_。运行时函数主要为芯片厂商提供实现算子函数时需要框架提供的工具函数，主要包括一些公共类型的声明以及分配与管理张量数据等；算子声明包含了用于人工智能计算的大量函数声明，为各个算子接口的具体参数及其类型提供了标准化的定义；C-API文档生成为算子声明生成API说明文档，供算子开发与使用者查阅

### 运行时函数(diopirt)
芯片厂商实现的算子函数时，计算过程中可能需要使用设备内存管理、流管理等runtime功能，这些都对设备资源的管理操作需要框架提供相关的函数实现。在proto中声明标准的运行时函数抽象，在算子函数运算需要分配内存或要在指定Stream上运行时，即可调用相关函数。
声明的内容主要包括以下部分：
-   错误码```diopiError_t```、数据类型 ```diopiDtype_t```、```diopiSize_t``` 以及不透明数据结构 ```diopiContextHandle_t``` 和 ```diopiTensorHandle_t```；
-   用于对 Tensor 对象进行操作的函数， 包括获取Tensor对象的内存、形状、类型等
-   用于对设备运行上下文Context进行操作的函数，包括获取Stream，构造Tensor对象等
-   其他函数：包括获取当前标准算子接口的版本

### 算子声明(functions)
目前实现的算子涵盖了卷积、归一化、池化、损失函数、基本代数运算、矩阵操作、数学运算等算子类型。

### C-API文档生成(doc)
使用标准算子接口进行训练框架和AI芯片的适配时，需要对算子函数的功能、参数及数据类型有详细的说明，以保证功能适配上的一致性。基于此需求，通过doxygen文档生成工具，生成对应C-API说明文档，作为算子开发与适配参考文档。

#### 依赖环境
生成文档依赖doxygen环境，安装命令如下：
```bash
yum install doxygen
```

#### **文档生成流程**
```
运行命令后可以在 docs 目录下看到生成的latex文档和html文档
```bash
cd docs && doxygen Doxyfile
```
