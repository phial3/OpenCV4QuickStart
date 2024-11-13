#!/bin/bash

# 遍历当前目录下的所有子目录
for dir in */; do
  # 检查是否是目录
  if [ -d "$dir" ]; then
    echo "进入目录: $dir"
    cd "$dir"
    
    # 检查是否存在Cargo.toml文件
    if [ -f "Cargo.toml" ]; then
      echo "执行 cargo clean"
      cargo clean
    else
      echo "未找到Cargo.toml，跳过"
    fi
    
    # 返回上一级目录
    cd ..
  fi
done

echo "所有操作完成。"
