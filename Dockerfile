# 1. 使用官方的 Miniconda3 基础镜像（自带 Conda）
FROM continuumio/miniconda3

# 2. 设置工作目录
WORKDIR /app

# 3. 把你的配置单复制进去
COPY environment.yml .

# 4. 根据配置单创建环境
# 这一步会自动下载并安装你原来环境里的所有库
RUN conda env create -f environment.yml

# --- 下面这两步是关键技巧 ---

# 5. 这里的 "my_project_env" 必须替换成你 environment.yml 文件第一行 "name:" 后面的名字
# 这一步是把新环境的路径加入系统 PATH，这样后续直接用 "python" 命令就是用的这个环境
ENV PATH /opt/conda/envs/neurostorm/bin:$PATH

# 6. 验证一下是否生效（可选，构建时会打印 Python 路径）
RUN echo "Conda environment path is: $PATH"

# --- 环境配好了，下面是你的代码 ---

# 7. 复制你的项目代码
COPY . .

# 8. 启动命令
CMD ["python", "app.py"]