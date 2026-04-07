#!/bin/bash
# 在容器内启动 Tensorboard（host 网络模式，直接通过 localhost:6006 访问）
# 注意：确保浏览器的代理设置绕过 127.0.0.1

docker exec -d agentic-rl tensorboard \
  --logdir=/workspace/agentic-rl/outputs \
  --bind_all \
  --port=6006

sleep 2
echo "Tensorboard 已启动: http://127.0.0.1:6006"
echo "如果打不开，检查浏览器代理设置是否绕过了 127.0.0.1"
