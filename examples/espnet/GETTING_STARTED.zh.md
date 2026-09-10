# ESPnet 音频语言模型使用指南

本文介绍 Bagpiper、OpusLM 和 OpusLM-dialogue 的模型转换、服务部署和客户端
调用。本仓库的默认分支为 `main`，基于 vLLM 0.28.0。项目总览见
[`README.md`](../../README.md)。

Bagpiper 使用场景描述生成音频。需要指定台词时，将台词放在场景描述的引号中，
例如：`A calm male voice says: 'Your package will arrive on Tuesday.'`。

---

## 名词表

| 词 | 意思 |
| --- | --- |
| codec | 把波形压成离散整数 token 的编解码器。这里出现两个：Xcodec（bagpiper 用）和 ESPnet DAC（opuslm 系列用）。 |
| stream（码本流） | 同一帧音频被多个码本各编一个 token。bagpiper 有 8 条流，opuslm 有 9 条。这些流在时间上对齐，逐帧并行。 |
| nq | opuslm 的流总数，等于 9：第 0 条是 XEUS 自监督特征经 kmeans 离散化的结果，第 1 到第 8 条是 DAC 的 8 个码本。 |
| XEUS | 一个自监督语音编码器。opuslm 系列用它把**输入**音频变成第 0 条流。 |
| task token | 一个特殊 token，放在 prompt 开头告诉 opuslm 这次要做什么。比如 82 是普通 TTS，80 是 ASR。 |
| CFG | classifier-free guidance。bagpiper 生成音频时可以开，服务端会自动配一条影子请求，两边的 logits 加权合并。开 CFG 会让这个请求的 KV cache 占用翻倍。 |
| V1 model runner | vLLM 里 `vllm/v1/worker/gpu_model_runner.py` 这个执行器。所有音频钩子都只写在它里面，所以必须用它，不能用 v0.28.0 默认的新执行器。 |
| 相位机（phase machine） | 一个跟着采样逐步推进的状态机，决定这一步该走文本相位还是音频相位。它在 model runner 里，不在模型里。 |

---

## 一、三个模型是什么，实现在哪

### bagpiper

主干是 Qwen3-8B，音频输入侧接 Qwen3-Omni 的 audio tower，音频输出侧接
Xcodec 解码器（HF tag `hf-audio/xcodec-hubert-general`，16 kHz）。8 条
codec 流，支持 CFG。词表 160392，其中 codec token 从 152192 开始。

它能做四件事：纯文本问答、听音频回答问题、文本转语音、文本转语音加 CFG。
注意它**没有**「音频进、音频出」这条路——`tts` 和 `tts_cfg` 不接受音频输入。

bagpiper 有一个需要提前知道的行为：它会先输出一段 Qwen3 风格的
`<think>` 推理，在推理里决定这次要回文本还是回音频，然后才动手。见第六节。

### opuslm

主干是 OLMo-2-7B。输入音频走 XEUS 加 kmeans，输出音频走 ESPnet DAC
（HF tag `ftshijt/espnet_codec_dac_large_v1.4_360epoch`）。两个 codec 都是
每帧 320 个采样点，16 kHz 下就是 50 Hz。9 条流。不支持 CFG。

任务由 task token 选：82 是普通 TTS，81 是带声音克隆的 TTS（额外给一段
参考音频），80 是 ASR，64 是纯文本续写。

### opuslm_dialogue

Llama 形状的 1.7B 主干，专门做对话。任务 token 89 是语音对话（用户说话，
助手用语音回答），88 是文本对话。可以额外给一段说话人参考音频来控制助手
音色。

### 实现落在哪些文件

模型层：

- `vllm/model_executor/models/bagpiper.py`
- `vllm/model_executor/models/opuslm.py`
- `vllm/model_executor/models/opuslm_dialogue.py`
- 三个模型在 `vllm/model_executor/models/registry.py` 注册。

配置层（HF config 的 Python 定义）：

- `vllm/transformers_utils/configs/bagpiper.py`
- `vllm/transformers_utils/configs/opuslm.py`
- `vllm/transformers_utils/configs/opuslm_dialogue.py`

分词器层：

- `vllm/tokenizers/opuslm.py`、`vllm/tokenizers/opuslm_dialogue.py`，在
  `vllm/tokenizers/registry.py` 注册。
- bagpiper 用标准的 HF Qwen3 分词器，所以这里没有 bagpiper 的文件。

引擎层（把逐流采样和音频输出接到 v1 引擎上）：

- `vllm/v1/worker/gpu_model_runner.py`：相位机、逐流采样、prefill 阶段的
  流回放、把 WAV 收集出来。
- `vllm/v1/core/sched/scheduler.py` 和 `vllm/v1/core/sched/utils.py`：
  opuslm 的 TTS 请求要推迟一次 EOS，否则音频段会被砍掉最后 9 步。
- `vllm/v1/engine/core.py`、`vllm/v1/outputs.py`、`vllm/outputs.py`、
  `vllm/v1/engine/output_processor.py`：音频从 model runner 一路传到
  `CompletionOutput.audio_output`。
- `vllm/entrypoints/openai/chat_completion/serving.py`：把音频挂到
  `ChatMessage.audio`，base64 编码的 16 kHz 单声道 PCM16 WAV。

工具和示例：

- `examples/espnet/convert/`：两个 checkpoint 转换脚本。
- `examples/espnet/serve_*.sh`：三个启动脚本。
- `examples/espnet/clients/`：四个参考客户端。
- `examples/espnet/docker/`：Dockerfile 和说明。

核心机制一句话：**只有第 0 条流进 vLLM 的序列和 KV cache**。第 1 到第
n 条流在模型的 `compute_logits` 里用一个私有的 top-k 采样器采出来，然后
以求和后的 embedding 形式回到下一步的输入。所以 vLLM 看到的序列长度就是
音频帧数，不是帧数乘以流数。

---

## 二、最短完整路径

### Checkpoint 格式

三个模型在 Hugging Face 上发布的都是**原生 ESPnet 权重**,没有 `config.json`、
没有 tokenizer、也没有 safetensors。`espnet/bagpiper-sft` 的 model card 自己就写
着「It is not a Transformers from_pretrained directory and no vLLM compatibility
is claimed」。所以**必须做一步参数转换**,这一节的 `convert/` 脚本就是干这个的。

| 本 fork 的名字 | 官方权重 | 发布格式 |
| --- | --- | --- |
| `bagpiper` | [`espnet/bagpiper-tts-sft`](https://huggingface.co/espnet/bagpiper-tts-sft)(做语音用这个) | `model.pt`,`{"module": state_dict}` |
| `opuslm` | [`espnet/OpusLM_7B_Anneal`](https://huggingface.co/espnet/OpusLM_7B_Anneal) | `model.pth` |
| `opuslm_dialogue` | [`espnet/multi_turn_SDS_RLAIF`](https://huggingface.co/espnet/multi_turn_SDS_RLAIF) | `2epoch.pth` |

官方 checkpoint 的来源、版本、哈希和转换规则见 [`MODELS.md`](MODELS.md)。
语音生成使用 `espnet/bagpiper-tts-sft`；当前集成不支持
`espnet/bagpiper-sft` 的音频输出。

转换脚本从固定 revision 的公开资源生成 `config.json` 和 tokenizer，并检查
资源的 SHA256。无需提供已有的模型配置目录。

### 0. 环境

推荐使用本仓库的 Docker 镜像，以保持 vLLM、CUDA、PyTorch 和 ESPnet
依赖一致。镜像支持 Linux amd64 和 arm64；GPU 功能检查覆盖 H100 和 GB200。
构建和启动命令见 [`docker/README.md`](docker/README.md)，依赖版本说明见
[`docker/COMPATIBILITY.md`](docker/COMPATIBILITY.md)。

以下转换命令在安装了本 fork 及 ESPnet 依赖的环境中，从仓库根目录执行。
镜像内的仓库路径为 `/workspace/vllm-fork`。启动脚本会检查 `PATH` 中的
`vllm` 是否来自本 fork。

### 1. 下载权重

使用上表中的 ESPnet 官方 checkpoint。各模型的下载和转换命令见下一节。
将 `MODEL_PATH` 指向转换后的模型目录，该目录应包含 `config.json`、
tokenizer 文件和 safetensors 权重。

音频处理还需要以下 codec 和 SSL 权重；首次加载时需要联网或预先填充缓存：

- Xcodec（bagpiper 用）：`hf-audio/xcodec-hubert-general`
- XEUS 和 kmeans（opuslm 系列的音频输入用）：`espnet/xeus`，一共约 2.3 GB
- ESPnet DAC（opuslm 系列的音频输出用）：
  `ftshijt/espnet_codec_dac_large_v1.4_360epoch`

前两个走标准的 HF hub 缓存（`~/.cache/huggingface`）。DAC 不走——
`espnet_model_zoo` 缓存在它自己的 site-packages 目录里。跑 Docker 时这两
个位置都要考虑，见第四节。

XEUS 还需要仓库里的 `model/config.yaml`，它不在默认下载清单里，实现已经
显式把它一起拉下来了。

### 2. 转换 checkpoint

从官方权重转，不需要别的输入。`config.json` 和 tokenizer 由脚本自己造，造完
先校验一遍，再开始写那 17 GB 权重 —— 顺序是故意这样排的，取源文件失败或者
哈希不对的时候不至于已经白写了一块盘。

**bagpiper**：输入是官方发布的 `model.pt`（也接受它由之的
`mp_rank_00_model_states.pt`），可以直接给下载目录。

```bash
hf download espnet/bagpiper-tts-sft --local-dir ~/ckpt/bagpiper-tts-sft
(cd ~/ckpt/bagpiper-tts-sft && sha256sum -c SHA256SUMS)

python examples/espnet/convert/convert_bagpiper_ckpt.py \
    ~/ckpt/bagpiper-tts-sft \
    ~/ckpt/bagpiper_converted
```

产物是 4 个 safetensors 分片加一个 index，约 17 GB。三处 `model_type` 都写成
新名字：顶层是 `bagpiper`，两个子 config 是 `bagpiper_audio_encoder` 和
`bagpiper_text` —— 这两个名字就是 `transformers_utils/configs/bagpiper.py`
里那两个 config 类自己声明的值。老的转换目录里留的还是 `speechlm_*`，那是改
名时没跟上的一半；那个字段实现根本不读，所以两种都能加载。

**opuslm**：输入是 ESPnet 的 `model.pth`。`--model` 必须给，脚本靠它决定去读
哪个官方 `config.yaml`、用哪个 backbone。

```bash
hf download espnet/OpusLM_7B_Anneal --local-dir ~/ckpt/opuslm
python examples/espnet/convert/convert_opuslm_ckpt.py \
    ~/ckpt/opuslm ~/ckpt/opuslm_converted --model opuslm
```

产物是 3 个分片加 index，约 14 GB。分片是因为 `--max-shard-size` 默认
5 GB，原始的 `model.pth` 是单个 14.8 GB 文件。

这个脚本还会**改写 chat template**。ESPnet 的 checkpoint 自带的模板是
`{% for message in messages %}{{'<|' + message['role'] + '|>' + message['content']}}{% endfor %}`，
它把每条消息包在 `<|role|>` 标记里。这些标记不是 opuslm 内层 BPE 词表的
条目，会被当成普通文本编码成好几个模型没见过的 token。脚本把它换成只取
内容的模板：

```text
{% for message in messages %}{{ message['content'] }}{% endfor %}
```

opuslm 真正需要的结构（`<sos/eos>`、task token、`<text_bpe_start/end>`、
ARDelay 的对齐 pad、`<codec_ssl_start/end>`）由 `OpusLMTokenizer` 在之后
加，加在 `text_token_start` 以下的保留 ID 区间里，位置正确。要换成别的
渲染方式，给 `vllm serve` 传 `--chat-template` 就能覆盖。

bagpiper 不需要这一步：它用标准 Qwen3 分词器，自带的
`chat_template.jinja` 直接可用。

**opuslm_dialogue 走同一条路，只是权重文件叫 `2epoch.pth`。**

```bash
hf download espnet/multi_turn_SDS_RLAIF --local-dir ~/ckpt/sds
python examples/espnet/convert/convert_opuslm_ckpt.py \
    ~/ckpt/sds ~/ckpt/opuslm_dialogue_converted --model opuslm_dialogue
```

这条分支是拿真实数据验过的：官方 `2epoch.pth` 里 221 张张量，按脚本的映射规则
处理之后丢掉 1 张 `criterion.*`，剩下 **220 张全部逐位相同**，最大绝对差 0.0，
形状和 dtype 两边一致。证明过程写在 [`MODELS.md`](MODELS.md)。

### 3. 起服务

三个脚本各自有默认端口，可以同时开在不同 GPU 上。

```bash
cd examples/espnet

CUDA_VISIBLE_DEVICES=0 MODEL_PATH=~/ckpt/bagpiper_converted \
    bash serve_bagpiper.sh                       # 端口 9811

CUDA_VISIBLE_DEVICES=1 MODEL_PATH=~/ckpt/opuslm_converted \
    bash serve_opuslm.sh                         # 端口 9812

CUDA_VISIBLE_DEVICES=2 MODEL_PATH=~/ckpt/opuslm_dialogue_converted \
    bash serve_opuslm_dialogue.sh                # 端口 9813
```

用 `PORT=xxxx` 换端口。脚本会在启动前检查端口是否被占，被占就直接报错。

脚本替你设好了两件必须的事：

- `VLLM_USE_V2_MODEL_RUNNER=0`。v0.28.0 默认把稠密模型路由到新的执行器
  `vllm/v1/worker/gpu/model_runner.py`，那里面一个音频钩子都没有。用了新
  执行器，服务能起来、能出文本，但会静默地不返回音频。所以脚本在检测到
  这个变量不是 0 时直接退出，宁可起不来也不给你一个半残的服务。
- `--no-async-scheduling`。v0.28.0 的异步调度默认开，脚本默认关，因为音
  频这条路上的验证绝大部分都是在关掉的配置下做的。想用上游默认值就设
  `ASYNC_SCHEDULING=1`，opuslm 的 TTS 在这个组合下实测过：两边的 token
  计数和输出 WAV 逐字节相同，见第三节。其余路径没有逐条在开启的配置下
  复测。

### 4. 调客户端

```bash
cd examples/espnet/clients
python client_bagpiper.py --task text --prompt "What is 2+2?"
```

四个客户端的完整用法在 `examples/espnet/clients/README.md`。下一节给几个
带实测结果的例子。

---

## 三、可以直接跑的推理示例

下面每个例子都在 H100 上真跑过，附的是当时的实际输出。采样带温度，你的
数字会有出入，但形状（有没有音频、多长、多少 token）应该一致。

需要输入音频的例子都用同一个文件，下面记作 `test.wav`。三个模型都会自己
把输入重采样到 16 kHz（bagpiper 按 feature extractor 声明的采样率，opuslm
系列写死 16 kHz），opuslm 和 opuslm_dialogue 还会自动混成单声道。所以采样
率不用你操心，但转成 16 kHz 单声道最省事：

```bash
ffmpeg -i 任意音频文件 -ar 16000 -ac 1 -c:a pcm_s16le test.wav
```

也可以使用 OpusLM TTS 生成一段 16 kHz 单声道 WAV 作为输入：

```bash
python client_opuslm.py --task tts --prompt "Hello world" --out test.wav
```

### bagpiper：纯文本

```bash
python client_bagpiper.py --task text --prompt "What is 2+2?"
```

实测：`finish_reason: stop`，257 个 completion token，答案正确，
`audio: (none)`。

### bagpiper：听音频回答问题

```bash
python client_bagpiper.py --task audio_understand \
    --audio test.wav --prompt "What sound is in this audio?"
```

实测：1386 个 completion token，返回一段详细描述，里面既有内容转写也有
音标转写，`audio: (none)`。

### bagpiper：生成音频

Bagpiper 根据自然语言场景描述生成音频。请求应描述声音、环境和说话方式，
并用引号指定台词。客户端默认 prompt 使用这一格式。

客户端内置的默认 prompt 已经是一个符合分布的例子,直接跑就行:

```bash
python client_bagpiper.py --task tts --out tts.wav
```

要让它说某一句特定的话,**描述场景,并把那句话放进引号**:

```bash
python client_bagpiper.py --task tts --out tts.wav \
    --prompt "A calm male voice, close-miked in a quiet studio, says: 'Your package will arrive on Tuesday.' No background noise."
```

示例音频、完整 prompt 和生成参数见 [`demo_assets/README.md`](demo_assets/README.md)。

### bagpiper：生成音频加 CFG

```bash
python client_bagpiper.py --task tts_cfg --cfg 3.0 --out tts_cfg.wav
```

同一条 prompt 开 CFG 3.0 的对照实测:1016 个 completion token、
`audio: 7.08s`(那条是较长的场景描述)。注意 CFG 会把两个分支的浮点差异
按 `main*3.0 + shadow*(1-3.0)` 放大约 7 倍,所以开 CFG 的输出天生比不开
更不可复现。

`--cfg 1.0` 等价于不开 CFG。开了之后服务端会自动建一条影子请求，这个请
求的 KV 占用翻倍。

### opuslm：TTS

```bash
python client_opuslm.py --task tts --prompt "Hello world" --out tts.wav
```

实测：`finish_reason: stop`，14 个 prompt token、68 个 completion token，
`audio: 1.16s @ 16000 Hz, 1 ch`。

68 这个数字可以自己验：58 步音频加 8 步 flush 加 2 步收尾。58 帧除以
50 Hz 是 1.16 秒，和 WAV 的时长一致。

这一条在两个互相独立的服务进程里各跑一次（一次开 async scheduling，一次
关），两份 WAV 逐字节相同，md5 都是 `9f4a1ee1225c1c8ef435e0ae82a49f0c`。
所以相同输入的音频输出是可复现的，尽管采样温度是 0.8 且没有传 `--seed`。

### opuslm：声音克隆 TTS

```bash
python client_opuslm.py --task tts --prompt "Hello world" \
    --audio test.wav --out tts_clone.wav
```

给了 `--audio` 就走 task token 81，参考音频决定音色。

### opuslm：ASR

```bash
python client_opuslm.py --task asr --audio test.wav
```

实测：810 个 prompt token、45 个 completion token，`audio: (none)`。输入是
一段 15.95 秒的英语朗读，转写出来的是：

```text
He heard first words I spoke in the original phonograph a little piece of
practical poetry. Mary had a little lamb, it reared quite a spell, and
everywhere that Mary went, the lamb was sure to go.
```

### opuslm：纯文本续写

```bash
python client_opuslm.py --task textlm --prompt "Once upon a time"
```

实测：6 个 prompt token、2048 个 completion token，`finish_reason: length`。
开头几句连贯（"the man in the moon got married. And now, in very tiny
letters, the wise sage of the land added…"），往后越来越散，最后退化成一个
词无限重复，一直撞到 token 上限才停。开着 tracer 能看到这个退化在 token
层面就是两个 id 在交替：`token=14508` 和 `token=13459` 反复出现。这是
checkpoint 本身的问题，不是这个 port 的缺陷，第六节有说明。

### opuslm_dialogue：语音对话

```bash
python client_opuslm_dialogue.py --task audio_dialogue \
    --audio test.wav --out reply.wav
```

实测：812 个 prompt token、764 个 completion token，`content` 是空串，
`audio: 15.08s @ 16000 Hz, 1 ch`。764 等于 754 步音频加 8 步 flush 加 2
步收尾，754 除以 50 Hz 正好 15.08 秒。

### opuslm_dialogue：文本对话

```bash
python client_opuslm_dialogue.py --task text_dialogue \
    --text "How are you today?" --out reply.wav
```

实测：19 个 prompt token、446 个 completion token，`finish_reason: stop`，
`audio: (none)`。

### 想看内部状态就开 tracer

```bash
VLLM_ESPNET_AUDIO_DEBUG=1 MODEL_PATH=... bash serve_opuslm.sh
```

日志里会多出三类前缀：`[espnet-audio]` 打相位机每步的迁移，
`[espnet-stop]` 打 EOS 推迟的判定，`[espnet-codec]` 打每条流的 token 是
否落在合法区间。

启动时会有一行 `WARNING ... Unknown vLLM environment variable detected:
VLLM_ESPNET_AUDIO_DEBUG`。这是 vLLM 的环境变量注册表不认识这个名字，
无害，可以忽略。

---

## 四、在个人 PC 上 build 和 run Docker

镜像基于 `vllm/vllm-openai:v0.28.0`，复用上游 CUDA 扩展，并安装本 fork
和 ESPnet 音频运行时。依赖与平台说明见 [`docker/README.md`](docker/README.md)。

### 两种架构:x86_64/amd64 与 ARM64/aarch64

`build.sh` 使用以下固定 digest 选择对应架构的基础镜像：

| 项 | 值 |
| --- | --- |
| index digest | `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` |
| `linux/amd64` | `sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635` |
| `linux/arm64` | `sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce` |

两个架构使用同一个 Dockerfile。构建脚本按 digest 固定基础镜像：

```bash
examples/espnet/docker/build.sh --arch amd64     # x86_64
examples/espnet/docker/build.sh --arch arm64     # aarch64
examples/espnet/docker/build.sh --arch both      # 多平台,输出 OCI layout
examples/espnet/docker/build.sh --arch amd64 --dry-run   # 只打印命令,不执行
```

跨架构构建需要先注册 QEMU binfmt
(`docker run --privileged --rm tonistiigi/binfmt --install all`);`--arch both`
还需要 buildx 的 container driver(`docker buildx create --use`)。多平台产物
无法 `--load` 进本地镜像库,所以 `--arch both` 写出一个 OCI layout 的 tar,
不往任何地方 push。

基础镜像按架构固定 digest。构建保留其中的 PyTorch/CUDA 组件，并安装带明确
版本后缀的 ESPnet 推理包；具体依赖处理见
[`docker/COMPATIBILITY.md`](docker/COMPATIBILITY.md)。依赖、模块导入和 CLI
检查都在镜像内执行；GPU 推理和音频效果需要单独验证。

不带脚本的等价写法(在仓库根目录构建,不是在 `examples/espnet/docker` 里,
解析出来的架构就是当前 daemon 所在的架构):

```bash
docker build -f examples/espnet/docker/Dockerfile -t espnet-vllm:v0.28.0 .
```

默认把当前工作树 `COPY` 进镜像。需要固定版本时，先 clone 仓库并 checkout
到指定 commit，再构建。

---

### 发布到 Docker Hub 的 espnet 组织

目标镜像仓库是 `espnet/vllm`。GitHub Actions 的
[`espnet-docker.yml`](../../.github/workflows/espnet-docker.yml) 负责取出 `main`
的代码、构建镜像、检查依赖和导入，再把通过检查的同一个镜像推送到 Docker Hub。
配置完成后，相关代码更新和每周定时任务都会触发它，也可以手动运行。
不需要额外开通 Docker Hub 的 Automated Builds。

organization owner 只需完成两件事：在 Docker Hub 的 `espnet` 下面创建公开
仓库 `vllm`，让现有 ESPnet CI 账号有推送权限；再让 GitHub 的 `espnet/vllm`
能使用 `DOCKERHUB_USERNAME` 和 `DOCKERHUB_TOKEN` 两项 secrets。如果已经是
organization secrets，只需把这个仓库加入可访问列表。另一个仓库的 environment
secrets 不会自动继承，无法从 GitHub 读回原 token。

构建使用 GitHub 提供的原生 x86 和 ARM 机器。无需自行提供服务器、注册 runner
或设置启用变量。两种架构检查都通过后才更新同一个 `latest`，拉取时 Docker 会
自动选择机器对应的架构。没有凭据时只构建和检查；填好后，下次相关 `main`
更新或每周一 03:23 UTC 的任务会自动发布，也可以手动运行 `publish=true`。

首次发布成功后可用 `docker pull espnet/vllm:latest`。需要固定版本时使用带
commit/run ID 的 tag 或镜像 digest。将 `ESPNET_DOCKER_PUBLISH=false` 设为
仓库变量可以暂停发布并保留构建检查。

凭据配置步骤见
[`docker/OWNER_SETUP.md`](docker/OWNER_SETUP.md)，完整流程见
[`docker/PUBLISHING.md`](docker/PUBLISHING.md)。仓库代码就绪不代表 Docker Hub
已经有公开镜像，首次推送仍需要上述凭据。

## 五、兼容性与验证

镜像内的依赖、模型导入和 CLI 检查由 GitHub Actions 执行。H100 / amd64 和
GB200 / arm64 的单 GPU 功能检查覆盖 Bagpiper 语音生成、OpusLM TTS / ASR，
以及 OpusLM-dialogue 的文本和语音对话。版本、参数和覆盖范围见
[`docker/VALIDATION.md`](docker/VALIDATION.md)。

默认配置使用 V1 model runner 和同步调度。多 GPU、吞吐性能和正式音质评测
不在上述容器功能检查的范围内。

Bagpiper 的抢占恢复会按绝对 token 位置回放其余音频流。完整与分块恢复均有
单 GPU 回归检查，但恢复后的波形不保证与无抢占生成逐样本一致。OpusLM 系列
的抢占恢复及全部并发 / CFG 组合尚未完成 GPU 验证。

---

## 六、常见问题与限制

### bagpiper 自己决定回文本还是回音频，system message 是那个开关

这是使用 bagpiper 最容易踩的一件事。在 `text_audio` 模式下，它先输出一段
`<think>` 推理，在里面判断这次该怎么回，然后可能：

- 输出文本段，接着输出 codec 帧（有音频）；
- 也可能在文本段之后直接停在 eos(2)，从头到尾不发 eot(3)。相位机是靠
  eot 从文本相位切到音频相位的，没有 eot 就没有音频段。这时请求返回没有
  音频是合法行为，不是 bug。

**决定它走哪条路的是 system message,而且只有一个正确的 system message。**
训练数据里每一条采样到的记录都带着同一段 system prompt(364 个字符,
md5 `903599715f6bea955adc1cfbf83aa9e2`),来源是
`bagpiper_sft/sft_part2/filtered_realistic.jsonl` 的 `system` 那一轮,4000 条
采样里一字不差地一致:

```text
You are a helpful assistant that generates audio based on user requests. You can
create various types of audio including sound effects, music, speech, ambient
sounds, and any combination of these. When given a request, first think through
what the user wants and how to create high-quality audio, then provide a detailed
description of the audio you will generate.
```

这段话本身就规定了「先想,再描述,然后渲染」这个输出契约 —— 也就是上面那个
`<think>` 加文本段加 codec 帧的顺序。所以 `client_bagpiper.py` 的 `tts` 和
`tts_cfg` 把 `--system` 默认设成它(常量 `DEFAULT_TTS_SYSTEM`),要关掉传
`--system ''`。

另外这两个任务的 `--max-tokens` 默认是 12000,和参考客户端一致
(`client_all.py` 第 33 行)——给小了,`<think>` 加文本段就把预算吃光了,
轮不到音频。

### opuslm 的 TTS 清晰度不达预期

已发布的 checkpoint 在 TTS 上不能可靠地把你给的文本念出来。音频出得来、
时长对得上、帧数算得通，但内容不一定是你要的那句话。

这一条排查过，**不是移植引入的缺陷**。逐一排除了七个独立的结构性原因之
后确认音频输出通路机械上是正确的：偏移量往返精确、逐流反交织精确、
`[espnet-codec]` 显示 8 条流的 token 全部落在合法区间（最后一次实测是
`outside=0/58`，8 条流全部如此）、相位统计与 token 数自洽、codec 在真实权
重上的往返通过。opuslm 的 textlm 和 opuslm_dialogue 的文本输出也有同样的
退化特征——开头连贯，往后变差——这指向 checkpoint 本身，不指向服务化。

textlm 那条路上这个退化最容易看清楚：开 tracer 跑
`--task textlm --prompt "Once upon a time"`，日志里会看到 `token=14508`
和 `token=13459` 两个 id 一直交替下去，直到撞上 token 上限。模型进了一个
两个 token 的循环，这是语言模型自身的退化，和 codec、相位机、EOS 推迟都
没有关系。

### 音频不支持流式

音频只在非流式响应里返回。整段生成完之后，`_collect_audio_outputs` 把
它收成一个 WAV，编码成 base64 放进 `ChatMessage.audio`。流式请求拿不到
音频。

### opuslm 的请求要在三个地方带上 mode

`mode` 这个字段必须同时出现在 `chat_template_kwargs`、
`mm_processor_kwargs` 和 `vllm_xargs` 三处。参考客户端已经这么做了，自己
拼 payload 的时候容易漏。

### 多轮语音对话的模态判定，模型和分词器规则不一致

模型侧用的是「这轮消息里**有没有**音频」（`any(...)`），分词器侧看的是
**最后一轮**是不是音频。单轮请求两者一致，多轮混合模态的请求下这两条规
则可能给出不同答案。目前只在单轮上验证过。

### audio_dialogue 的 prompt 末尾那个 35 不起作用

`audio_dialogue` 的 prompt 末尾会带一个 `35`（`text_bpe_start_end`），但
它不生效——model runner 给这个任务播的初始相位就是 `audio`，直接跳过了
文本相位。不影响结果，知道就行，免得对着 prompt 找不着对应的行为。

### 必须用 V1 model runner

前面说过一次，这里再记一遍，因为它的失败方式最阴：用了 v0.28.0 默认的新
执行器，服务能正常启动、能正常出文本，只是永远不返回音频，日志里没有报
错。启动脚本靠硬性检查把这个情况堵在启动之前。

### CFG 让 KV 占用翻倍

开 CFG 的请求，服务端会额外建一条影子请求。压测时按两倍算 KV 预算。

### 那个 "Unknown vLLM environment variable" 警告

`VLLM_ESPNET_AUDIO_DEBUG` 不在 vLLM 的环境变量注册表里，所以启动时会警告
一次。功能正常，警告可以忽略。
