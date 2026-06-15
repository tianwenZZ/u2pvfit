* Nightly-build里，Rec run5强依赖x86_64_v3-el9-gcc15-opt，用gcc13会报错

* 但我们的Allen需要在gcc13 (x86_64_v3-el9-gcc13+cuda12_4-opt+g)上编，如果用gcc15，后续支持会出问题
  * ROOT 在 gcc15 平台上是用 C++23 编译的（ROOT_CXX_STANDARD=23），Gaudi 和 Allen 都从 ROOT 继承了这个标准，我们加的 -DCMAKE_CXX_STANDARD=20 被覆盖了。
  * nvcc13.0不支持cpp23


# Allen 编译

* 如果对应的 build dir **还没有 configure 过**，比如你第一次编 CPU：
```bash
make BINARY_TAG=x86_64_v3-el9-gcc13-opt+g Allen
```
它会读最新的 `utils/config.json`，生成：
```text
Allen/build.x86_64_v3-el9-gcc13-opt+g
```
并用当前 config 里的 `cmakeFlags`。
但如果这个 build dir **已经 configure 过**，比如已经有：
```text
Allen/build.x86_64_v3-el9-gcc13-opt+g/build.ninja
```
那直接：
```bash
make BINARY_TAG=x86_64_v3-el9-gcc13-opt+g Allen
```
通常只会增量编译，不一定重新应用新的 `cmakeFlags`。这时要显式 reconfigure：
```bash
make BINARY_TAG=x86_64_v3-el9-gcc13-opt+g Allen/configure
make BINARY_TAG=x86_64_v3-el9-gcc13-opt+g Allen
```
或者更完整：
```bash
make BINARY_TAG=x86_64_v3-el9-gcc13-opt+g Allen/install
```
如果你**只是改了 `.cu/.cuh/.py` 源码**，直接 `make Allen` 就行。  
如果你**改的是 `utils/config.json` 里**的 `binaryTag`、`cmakeFlags`、`Allen_DIR` 这种 configure 相关设置，建议跑 `Allen/configure`。

* **如果只改了cuda文件**: 只需要重新`make fast/Allen`
* **GPU上以单线程配置编译Allen**：`utils/config.json`里`"cmakeFlags"`的`"Allen"`加 `-DCMAKE_CUDA_FLAGS=-DALLEN_DEBUG_SINGLE_THREAD_PV`
* **手动source Allen环境：**
```
cd /etude/tzhou/workdir/stack-run5cpu-260515
Allen/build.x86_64_v3-el9-gcc13+cuda12_4-opt+g/run <command>  
Allen/build.x86_64_v3-el9-gcc13-opt+g/run <command>  
echo $BINARY_TAG
```
同一份源码，分别编：
```
cd /etude/tzhou/workdir/stack-run5gpu-260515
make BINARY_TAG=x86_64_v3-el9-gcc13+cuda12_4-opt+g Allen
make BINARY_TAG=x86_64_v3-el9-gcc13-opt+g Allen
```

然后分别跑：`source Allen/build.x86_64_v3-el9-gcc13+cuda12_4-opt+g/allenenv.sh