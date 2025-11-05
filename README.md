# GME Service

## 项目简介

GME Service 提供一个基于 FastAPI 的服务，用于调用 ModelScope 的 `iic/gme-Qwen2-VL-2B-Instruct` 模型生成多模态向量表征，支持：
- 文本嵌入：可选提示词（instruction），返回文本的向量列表。
- 图像嵌入：接收上传的单张图片文件，返回对应的向量。

首次运行时服务会自动下载模型权重，并在检测到 GPU 时优先使用 GPU 加速推理。

## 环境准备（conda）

1. 创建并激活 conda 环境（Python 3.11 及以上）：

   ```bash
   conda create -n gme-service python=3.11
   conda activate gme-service
   ```

2. 安装依赖：

   ```bash
   pip install fastapi uvicorn torch Pillow modelscope packaging
   pip install transformers==4.51.3 peft diffusers
   ```

> 如需 GPU 支持，请根据实际 CUDA 版本安装匹配的 `torch`。

## 本地运行

```bash
uvicorn gme_service:app --host 0.0.0.0 --port 8000
```

服务启动后提供两个 REST 接口：`/embed/text` 与 `/embed/image`。

## API 说明

- `POST /embed/text`  
  请求体示例：
  ```json
  {
    "texts": ["示例句子"],
    "prompt": "可选的指令"
  }
  ```
  返回：每个文本对应的嵌入向量数组。

- `POST /embed/image`  
  使用 multipart/form-data 上传单张图片，字段名为 `file`。返回该图片的嵌入向量。

## 模型缓存位置

服务在启动时会调用 `snapshot_download` 将模型下载到 `GME_MODEL_DIR` 环境变量指定的目录。若未设置该变量，则默认写入 `~/gme_models`，并在其中创建 `iic/gme-Qwen2-VL-2B-Instruct` 子目录保存全部模型文件。下次启动会直接从该目录加载，无需重新下载。

## 多 GPU 使用

服务会在启动时检测 `torch.cuda.device_count()`：若可用 GPU 数量大于 1，会自动通过 `torch.nn.DataParallel` 对模型进行包装，实现多卡并行推理。确保在部署前正确设置 `CUDA_VISIBLE_DEVICES`（例如 `export CUDA_VISIBLE_DEVICES=0,1`），即可让两张 RTX 3090 同时工作。如需自定义并行策略，可在 `gme_service.py` 中调整相关逻辑。

## 在 Linux 服务器上使用 systemd 部署

1. 将项目放置在如 `/opt/gme_service` 的目录中，并使用 conda 创建环境（假设 conda 安装在 `/opt/conda`）：

   ```bash
   conda create -n gme_service python=3.11
   conda activate gme_service
   pip install fastapi uvicorn torch Pillow modelscope
   ```

   安装完成后可运行 `conda deactivate`；systemd 将直接调用虚拟环境中的可执行文件。

2. 创建 systemd Unit `/etc/systemd/system/gme_service.service`：

   ```ini
   [Unit]
   Description=GME Embedding Service
   After=network.target

   [Service]
   Type=simple
   User=gme
   Group=gme
   WorkingDirectory=/opt/gme_service
   Environment="PATH=/opt/conda/envs/gme_service/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
   Environment="CUDA_VISIBLE_DEVICES=0,1"
   Environment="TOKENIZERS_PARALLELISM=false"
   Environment="GME_MODEL_DIR=/opt/gme_service/models"
   ExecStart=/opt/conda/envs/gme_service/bin/uvicorn gme_service:app --host 0.0.0.0 --port 8000
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

   - 根据实际情况调整 `User`、`Group`（推荐使用专用服务账号）、`WorkingDirectory`、端口号等。
   - 如果 conda 安装位置不同，请修改 `PATH` 中的前缀，使其指向目标环境的 `bin` 目录。
   - `CUDA_VISIBLE_DEVICES` 控制可使用的 GPU 编号，可按需调整。
   - `GME_MODEL_DIR` 用于指定模型落盘目录，便于持久化管理与备份。
   - 可以通过以下命令直接在命令行生成该 unit 文件：

     ```bash
     sudo tee /etc/systemd/system/gme_service.service > /dev/null <<'EOF'
     [Unit]
     Description=GME Embedding Service
     After=network.target

     [Service]
     Type=simple
     User=gme
     Group=gme
     WorkingDirectory=/opt/gme_service
     Environment="PATH=/opt/conda/envs/gme_service/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
     Environment="CUDA_VISIBLE_DEVICES=0,1"
     Environment="TOKENIZERS_PARALLELISM=false"
     Environment="GME_MODEL_DIR=/opt/gme_service/models"
     ExecStart=/opt/conda/envs/gme_service/bin/uvicorn gme_service:app --host 0.0.0.0 --port 8000
     Restart=always
     RestartSec=5

     [Install]
     WantedBy=multi-user.target
     EOF
     ```

3. 重载 systemd 并设置开机自启、启动服务：

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable gme_service
   sudo systemctl start gme_service
   ```

4. 常用运维命令：

   ```bash
   sudo systemctl status gme_service      # 查看运行状态
   sudo journalctl -u gme_service -f      # 实时查看日志
   sudo systemctl restart gme_service     # 重启服务
   sudo systemctl stop gme_service        # 停止服务
   ```
