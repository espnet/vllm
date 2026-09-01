# 在 vLLM 上跑 bagpiper / opuslm / opuslm_dialogue

这份文档讲清一件事：怎么在一台 H100 机器上，把这三个语音语言模型从
checkpoint 变成一个能收 HTTP 请求、会返回音频的服务。读它不需要知道
这些模型以前在哪个 vLLM 分支上跑过。

代码在分支 `espnet-audio-v0.28.0` 上，基线是上游 tag `v0.28.0`
（commit `2cf0a6915c`）。

---

## 名词表

先把后面反复出现的词定义清楚。

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

主干是 Olmo3。输入音频走 XEUS 加 kmeans，输出音频走 ESPnet DAC
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

### 0. 环境

要求：Linux、NVIDIA H100（sm90）、CUDA 13 驱动、Python 3.12。

```bash
python3 -m venv ~/venvs/espnet-vllm
source ~/venvs/espnet-vllm/bin/activate

# 装这个 fork。VLLM_USE_PRECOMPILED=1 让它复用官方 v0.28.0 wheel 里
# 已经编好的 CUDA kernel，几分钟装完；不加这个变量要从源码编译几小时。
cd /path/to/this/repo
VLLM_USE_PRECOMPILED=1 pip install -e .

# 音频必须的两个包。espnet 提供 codec 的实现，espnet_model_zoo 负责
# 解析 DAC 的 model tag——只装 espnet 的话，第一次解码音频会抛
# ModuleNotFoundError。
pip install espnet espnet_model_zoo
```

如果要用音频**输入**（bagpiper 的 audio_understand、opuslm 的 ASR 和声音
克隆、opuslm_dialogue 的语音对话），还需要 `joblib` 和 `scikit-learn` 来
加载 XEUS 的 kmeans 模型。`espnet` 一般会连带装上，没有就补装。

**PATH 要检查一遍。** 如果机器上另有一份系统 vllm（比如
`/usr/local/bin/vllm`），它不认识这三个模型，服务会以 model type 未知
失败。启动脚本里有一道前置检查会在这种情况下直接报错退出，不会让你跑起
一个半残的服务。把 venv 的 bin 放在 PATH 最前面就行：

```bash
export PATH=~/venvs/espnet-vllm/bin:$PATH
```

### 1. 下权重

三个模型的权重都在 HF 仓库 `anonymous-release/vLLM_alm` 的 `main` 分支
上，分别在 `bagpiper/`、`OpusLM/`、`OpusLM_dialogue/` 三个子目录里。

```bash
# bagpiper：一个 DeepSpeed 分片 checkpoint 加上一整套 tokenizer 文件
hf download anonymous-release/vLLM_alm --include "bagpiper/*" \
    --local-dir ~/ckpt/vLLM_alm

# opuslm：ESPnet 的 model.pth 加 tokenizer
hf download anonymous-release/vLLM_alm --include "OpusLM/*" \
    --local-dir ~/ckpt/vLLM_alm

# opuslm_dialogue：直接就是 safetensors，不用转换
hf download anonymous-release/vLLM_alm --include "OpusLM_dialogue/*" \
    --local-dir ~/ckpt/vLLM_alm
```

**注意下载出来的目录会多一层。** `--local-dir ~/ckpt/vLLM_alm` 加
`--include "OpusLM_dialogue/*"` 得到的模型目录是
`~/ckpt/vLLM_alm/OpusLM_dialogue`，`MODEL_PATH` 要指到这一层，指到
`~/ckpt/vLLM_alm` 会启动失败。

codec 和 SSL 的权重会在**服务启动时**从 HF 拉取，不是等到第一个请求：

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

**bagpiper**：输入是 DeepSpeed 的 `mp_rank_00_model_states.pt`，
`--ref-dir` 指向同一个下载目录，脚本从那里拷 config 和 tokenizer。

```bash
cd examples/espnet/convert
python convert_bagpiper_ckpt.py \
    ~/ckpt/vLLM_alm/bagpiper/mp_rank_00_model_states.pt \
    ~/ckpt/bagpiper_converted \
    --ref-dir ~/ckpt/vLLM_alm/bagpiper
```

产物是 4 个 safetensors 分片加一个 index，约 17 GB。脚本会把顶层
`config.json` 的 `model_type` 改写成 `bagpiper`；两个子 config 的
`model_type` 保持 `speechlm_audio_encoder` 和 `speechlm_text` 不变，那是
checkpoint 里原本的名字，实现按这个名字读。

**opuslm**：输入是 ESPnet 的 `model.pth`。

```bash
python convert_opuslm_ckpt.py \
    ~/ckpt/vLLM_alm/OpusLM/model.pth \
    ~/ckpt/opuslm_converted \
    --ref-dir ~/ckpt/vLLM_alm/OpusLM
```

产物是 3 个分片加 index，约 14 GB。分片是因为 `--max-shard-size` 默认
5 GB，原始的 `model.pth` 是单个 14.8 GB 文件。

这个脚本还会**改写 chat template**。ESPnet 的 checkpoint 自带的模板是
`{% for message in messages %}{{'<|' + message['role'] + '|>' + message['content']}}{% endfor %}`，
它把每条消息包在 `<|role|>` 标记里。这些标记不是 opuslm 内层 BPE 词表的
条目，会被当成普通文本编码成好几个模型没见过的 token。脚本把它换成只取
内容的模板：

```
{% for message in messages %}{{ message['content'] }}{% endfor %}
```

opuslm 真正需要的结构（`<sos/eos>`、task token、`<text_bpe_start/end>`、
ARDelay 的对齐 pad、`<codec_ssl_start/end>`）由 `OpusLMTokenizer` 在之后
加，加在 `text_token_start` 以下的保留 ID 区间里，位置正确。要换成别的
渲染方式，给 `vllm serve` 传 `--chat-template` 就能覆盖。

bagpiper 不需要这一步：它用标准 Qwen3 分词器，自带的
`chat_template.jinja` 直接可用。

**opuslm_dialogue 不需要转换。** 它下载下来就是 `model.safetensors`，
`config.json` 里已经写着 `model_type: opuslm_dialogue`，
`tokenizer_config.json` 里的 chat template 已经是只取内容的那个。直接把
`MODEL_PATH` 指到下载目录就能起服务。也正因为它不带 `model.pth`，
`convert_opuslm_ckpt.py` 处理 dialogue 的那条分支没有真实数据可测。

### 3. 起服务

三个脚本各自有默认端口，可以同时开在不同 GPU 上。

```bash
cd examples/espnet

CUDA_VISIBLE_DEVICES=0 MODEL_PATH=~/ckpt/bagpiper_converted \
    bash serve_bagpiper.sh                       # 端口 9811

CUDA_VISIBLE_DEVICES=1 MODEL_PATH=~/ckpt/opuslm_converted \
    bash serve_opuslm.sh                         # 端口 9812

CUDA_VISIBLE_DEVICES=2 MODEL_PATH=~/ckpt/vLLM_alm/OpusLM_dialogue \
    bash serve_opuslm_dialogue.sh                # 端口 9813
```

用 `PORT=xxxx` 换端口。脚本会在启动前检查端口是否被占，被占就直接报错。

脚本替你设好了两件必须的事：

- `VLLM_USE_V2_MODEL_RUNNER=0`。v0.28.0 默认把稠密模型路由到新的执行器
  `vllm/v1/worker/gpu/model_runner.py`，那里面一个音频钩子都没有。用了新
  执行器，服务能起来、能出文本，但会静默地不返回音频。所以脚本在检测到
  这个变量不是 0 时直接退出，宁可起不来也不给你一个半残的服务。
- `--no-async-scheduling`。v0.28.0 的异步调度默认开，脚本默认关，因为音
  频这条路是在关掉的配置下验证的。想用上游默认值就设
  `ASYNC_SCHEDULING=1`（这个组合没有实测过）。

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

我实测用的那段就是这样从一个 ogg 转出来的，长约 16 秒、510,380 字节，内容
是一段清晰的英语朗读。手边没有素材的话，用 opuslm 自己的 TTS 先合成一段也
可以，它的输出正好就是 16 kHz 单声道 WAV：

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

### bagpiper：文本转语音

```bash
python client_bagpiper.py --task tts \
    --prompt "Read this aloud in a calm voice: The quick brown fox jumps over the lazy dog." \
    --out tts.wav
```

实测：788 个 completion token，`audio: 4.04s @ 16000 Hz, 1 ch`，
`tts.wav` 129,324 字节。

有意思的是模型在 `<think>` 里会给自己定一个时长目标（那次写的是
"Generate a 3.9-second audio clip"），渲染出来的 4.04 秒对得上。

### bagpiper：文本转语音加 CFG

```bash
python client_bagpiper.py --task tts_cfg --cfg 3.0 \
    --prompt "A dog barking twice in a quiet room." --out tts_cfg.wav
```

实测：702 个 completion token，`audio: 3.20s @ 16000 Hz, 1 ch`，
102,444 字节。

`--cfg 1.0` 等价于不开 CFG。开了之后服务端会自动建一条影子请求，这个请
求的 KV 占用翻倍。

### opuslm：TTS

```bash
python client_opuslm.py --task tts --prompt "Hello world" --out tts.wav
```

实测：`finish_reason: stop`，22 个 prompt token、281 个 completion token，
`audio: 5.42s @ 16000 Hz, 1 ch`。

281 这个数字可以自己验：271 步音频加 8 步 flush 加 2 步收尾。271 帧除以
50 Hz 是 5.42 秒，和 WAV 的时长一致。

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

实测：53 个 completion token，转写准确，`audio: (none)`。

### opuslm：纯文本续写

```bash
python client_opuslm.py --task textlm --prompt "Once upon a time"
```

实测：277 个 completion token。开头连贯，往后会退化，见第六节。

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

**这个镜像没有在当前集群上构建过，也没有推到任何 registry。** 集群不允许
build 镜像，也不允许 push。所以下面的命令是按 Dockerfile 的内容写的，只
做过静态和配置层面的核对，没有实际构建验证。这是本文档里唯一一处未经端到
端验证的部分。

镜像从上游官方镜像 `vllm/vllm-openai:v0.28.0` 派生，H100 属于 sm90，上游
wheel 已经覆盖，所以不需要本地编译 CUDA kernel。

在仓库根目录构建（不是在 `examples/espnet/docker` 里）：

```bash
docker build -f examples/espnet/docker/Dockerfile -t espnet-vllm:v0.28.0 .
```

默认这条路径把你当前的工作树 `COPY` 进镜像，所以镜像里装的就是你手上这份
代码。如果想改成从 git clone 拉，需要动 Dockerfile 本身：把第 33 行那句
`COPY . /workspace/vllm-fork` 换成 Dockerfile 注释里给出的 `ARG` 加
`RUN git clone` 三行，然后才能用 `--build-arg VLLM_FORK_URL=...` 传参。光
传 build-arg 不改文件是没有效果的。

运行：

```bash
docker run --gpus all --rm -p 9811:9811 \
    -v /path/to/checkpoints:/models \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    espnet-vllm:v0.28.0 \
    /models/bagpiper_converted \
    --served-model-name bagpiper --port 9811 \
    --trust-remote-code --max-model-len 16384 \
    --limit-mm-per-prompt '{"audio": 1}' --no-async-scheduling
```

镜像保留了上游的 `ENTRYPOINT ["vllm", "serve"]`，所以 `docker run` 后面
第一个参数就是模型路径。

要用的东西：

- **挂载 HF 缓存**（`-v ~/.cache/huggingface:/root/.cache/huggingface`）。
  Xcodec 和 XEUS 走这个缓存，XEUS 有 2.2 GB，不挂就每次容器启动重新下。
- **DAC 已经预热进镜像**，占 308 MB。为什么单独处理：
  `espnet_model_zoo` 缓存在它自己的 site-packages 目录里，不在 HF 缓存
  里，挂 HF 缓存管不到它。
- `VLLM_USE_V2_MODEL_RUNNER=0` 用 `ENV` 写进镜像了，不用自己传。
- 端口按模型分：bagpiper 9811，opuslm 9812，opuslm_dialogue 9813。

想在容器里用启动脚本而不是直接 `vllm serve`，覆盖 entrypoint：

```bash
docker run --gpus all --rm -p 9811:9811 \
    -v /path/to/checkpoints:/models \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash espnet-vllm:v0.28.0 \
    -c 'MODEL_PATH=/models/bagpiper_converted bash /workspace/vllm-fork/examples/espnet/serve_bagpiper.sh'
```

更多细节在 `examples/espnet/docker/README.md`。

---

## 五、验证到了哪一步

平台：8×H100 80GB HBM3，Linux，overlay 文件系统。

版本：Python 3.12.3，vLLM 0.28.0+precompiled（editable 安装），
torch 2.13.0+cu130，transformers 5.16.1，espnet 202511，
espnet-model-zoo 0.1.7，librosa 1.0.0，soundfile 0.14.0，
scikit-learn 1.9.0，joblib 1.6.0。

**转换**：

| 模型 | 输入 | 结果 |
| --- | --- | --- |
| bagpiper | 真实的 `mp_rank_00_model_states.pt` | 转出 4 分片共 17 GB，服务起得来，四个任务全通 |
| opuslm | 真实的 `model.pth` | 转出 3 分片共 14 GB，键集与参考完全一致（`KEYSETS_EQUAL`），逐张量比对 356 个张量零不匹配、最大绝对差 0.0 |
| opuslm_dialogue | 不需要转换 | 直接从下载目录起服务 |

chat template 的改写单独验过：把参考的 `tokenizer_config.json.orig` 拿
出来，跑一遍脚本里的 `fix_chat_template`，产物与已验证服务实际加载的那份
逐字段一致（模板一致、键集一致、其他字段零差异）。

**推理**：三个模型九条路径全部在 H100 上真实跑过，结果见第三节。用的是
仓库里的参考客户端，不是绕过客户端的私有脚本。

**其他实测过的点**：

- 两个 codec 在真实权重上的编解码往返（CPU）通过。
- 偏移量往返和逐流反交织精确无误
  （`OFFSET_ROUNDTRIP_EXACT`、`DEINTERLEAVE_STREAM0_EXACT`、
  `DEINTERLEAVE_STREAM18_EXACT`）。
- DAC 预热进镜像的步骤通过（`PREWARM_OK`，308 MB）。
- 启动脚本的前置检查双向验过：认得出这个 fork，也拒绝得掉系统 vllm。
- 音频输出对同一输入是可复现的。同一段输入音频在两个独立的服务进程里跑
  `audio_dialogue`，得到的 WAV 字节完全相同（md5 都是
  `f4d0640e406ee55c7987c5cfc414c015`）。换一段输入音频，结果就变
  （md5 `5cfdc5135b97fc91672885756523460b`，时长 8.02 秒对 15.08 秒）。
  所以这是可复现的采样，不是写死的输出。

**测试**：单元测试只能在装了 transformers 5 的环境里跑。本地机器上是
transformers 4，`vllm/transformers_utils/config.py` 会直接抛
`ImportError`，所以全部 pytest 都在 H100 机器的那个 venv 里跑。

针对这三个模型新增的五个测试文件（`tests/model_executor/test_bagpiper.py`、
`tests/model_executor/test_opuslm.py`、`tests/tokenizers_/test_opuslm.py`、
`tests/tokenizers_/test_registry.py`、`tests/v1/core/test_opuslm_stop.py`）
在最后一个代码提交上复跑过，结果是 `38 passed, 16 warnings in 12.90s`。

除了这五个文件，还跑了一轮更宽的回归，覆盖 `tests/v1/core`、
`tests/tokenizers_`、`tests/model_executor`、`tests/transformers_utils`：

```
1 failed, 1142 passed, 223 skipped, 16 warnings, 94 errors in 1476.73s (0:24:36)
```

那 95 条不通过的用例，全部与本分支无关，逐条查过来源：

- 94 个 error 里有 92 个来自 `tests/tokenizers_/test_detokenize.py`，它按
  `meta-llama/Llama-3.2-1B-Instruct` 和 `mistralai/Pixtral-12B-2409` 做参
  数化。日志里报 `GatedRepoError: 401` 和
  `Cannot access gated repo`，原因是这台机器上的 HF token 没有这两个受限
  仓库的访问权。
- 剩下 2 个 error（`tests/v1/core/test_scheduler_e2e.py` 的
  `test_concurrent_partial_prefill` 和 `test_prefix_cache_stats_is_recorded`）
  加上唯一那 1 个 failed（`tests/v1/core/test_reset_prefix_cache_e2e.py::test_reset_prefix_cache_e2e`）
  报的是同一件事：`Free memory on device cuda:0 (9.92/79.18 GiB) on startup
  is less than desired GPU memory utilization`。那一轮回归跑的时候，我自己
  的三个服务正占着 GPU 0 到 2。这两个文件本分支一行都没改过
  （`git diff --stat v0.28.0..HEAD` 对这两个路径为空）。

为了不把这条留成推断，我把三个服务停掉、确认 8 张卡都回到 0 MiB，再在空闲
GPU 上单独复跑这两个文件，结果是 `3 passed, 15 warnings in 57.52s`。所以那
三条确实是显存占用导致的，不是代码缺陷。

---

## 六、常见问题与限制

### bagpiper 自己决定回文本还是回音频，system message 是那个开关

这是使用 bagpiper 最容易踩的一件事。在 `text_audio` 模式下，它先输出一段
`<think>` 推理，在里面判断这次该怎么回，然后可能：

- 输出文本段，接着输出 codec 帧（有音频）；
- 也可能在文本段之后直接停在 eos(2)，从头到尾不发 eot(3)。相位机是靠
  eot 从文本相位切到音频相位的，没有 eot 就没有音频段。这时请求返回没有
  音频是合法行为，不是 bug。

**决定它走哪条路的是 system message。** 实测数据：同一个 prompt、同样的
采样参数、`max_tokens: 12000`，带
`{"role": "system", "content": "You are a helpful assistant."}` 的 8 个
请求里 7 个出音频，不带的 8 个请求里 0 个出音频（Fisher 精确检验
p≈0.0014）。16 个响应的 `<think>` 块全都完整闭合，所以推理本身不是那个
区分因素。

prompt 的写法是个弱得多的杠杆：不带 system message 时，四种不同写法
（指令加句子、光句子、"Say: ..."、声音描述）一共 16 个请求，出音频 0 个。

所以 `client_bagpiper.py` 的 `tts` 和 `tts_cfg` 把 `--system` 默认设成了
`You are a helpful assistant.`，这也是上游参考客户端每个请求都在发的东西。
要关掉传 `--system ''`。另外这两个任务的 `--max-tokens` 默认是 12000，和
参考客户端一致——给小了，`<think>` 加文本段就把预算吃光了，轮不到音频。

### opuslm 的 TTS 清晰度不达预期

已发布的 checkpoint 在 TTS 上不能可靠地把你给的文本念出来。音频出得来、
时长对得上、帧数算得通，但内容不一定是你要的那句话。

这一条排查过，**不是移植引入的缺陷**。逐一排除了七个独立的结构性原因之
后确认音频输出通路机械上是正确的：偏移量往返精确、逐流反交织精确、
`[espnet-codec]` 显示 8 条流的 token 全部落在合法区间（`outside=0/271`）、
相位统计与 token 数自洽、codec 在真实权重上的往返通过。opuslm 的 textlm
和 opuslm_dialogue 的文本输出也有同样的退化特征——开头连贯，往后变差——
这指向 checkpoint 本身，不指向服务化。

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
