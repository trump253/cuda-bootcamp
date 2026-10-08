# 容器内使用 Nsight Systems GUI

本指南针对当前 Ubuntu 20.04 容器、Nsight Systems 2022.4.2，以及 SSH / VS Code Remote SSH 连接方式。2026-10-08 已补齐动态库并启动 Xvfb、Openbox、x11vnc 和 noVNC；当前会话先从下面的端口转发开始。

## 在本地看到界面

使用 VS Code Remote SSH 时，打开“端口 / Ports”面板，选择“转发端口”，输入远程端口 `6080`。在本地浏览器打开转发后的地址，并访问 `/vnc.html`，点击“连接 / Connect”。如果 VS Code 分配的本地端口也是 `6080`，地址就是 `http://127.0.0.1:6080/vnc.html`。

使用普通 SSH 时，在**本地电脑的终端**运行下面的命令，把 `your-container` 换成平时连接这个容器的 SSH 别名或目标；需要特殊 SSH 端口或跳板机时沿用原来的参数。

```bash
ssh -N -L 6080:127.0.0.1:6080 your-container
```

保持隧道运行，在本地浏览器打开 `http://127.0.0.1:6080/vnc.html` 并点击“连接”。转发目标应是当前容器的网络环境。noVNC 与 VNC 只监听容器的回环地址，通过 SSH 访问。

## 打开与观察报告

在 GUI 中使用 `File → Open` 打开当前仓库 `build/` 下的 `.nsys-rep`。也可以在容器终端设置本次会话的显示变量后，直接运行熟悉的 `nsys-ui` 命令：

```bash
export DISPLAY=:99
export LIBGL_ALWAYS_SOFTWARE=1
export __GLX_VENDOR_LIBRARY_NAME=mesa
export QTWEBENGINE_CHROMIUM_FLAGS=--no-sandbox
nsys-ui build/day10_vector_add.nsys-rep
```

这些变量只作用于当前终端。`DISPLAY` 指向虚拟屏幕；Mesa 的 llvmpipe 在 CPU 上渲染界面，本次实测 OpenGL 3.1；`QTWEBENGINE_CHROMIUM_FLAGS` 处理 root 运行 Qt WebEngine 的启动要求。

打开报告后，展开 GPU 下的 CUDA HW 行查看 kernel 与内存传输，再查看进程/线程下的 CUDA API 行。先缩放到第二组较大的 Vector Add 用例，按 [README.md](README.md) 的四个问题记录 CPU 调用、GPU 执行和同步。启动 GUI 的环境验证不替代 Day 10 的时间线分析验收。

## 容器重启后重新启动

当前组件已经运行。以下步骤用于显示服务退出后重建；重新创建容器时还需要再次安装依赖。命令从仓库根目录执行。

```bash
apt-get install -y --no-install-recommends \
  libopengl0 libegl1 libegl-mesa0 libnss3 libnspr4 libjpeg62 \
  xvfb mesa-utils x11-utils xauth openbox x11vnc novnc websockify
```

启动虚拟显示并确认它已经就绪，再启动桌面与浏览器转发。日志和 PID 写到被 Git 忽略的 `build/nsys-gui/`。

```bash
mkdir -p build/nsys-gui
nohup /usr/bin/Xvfb :99 -screen 0 1600x900x24 -nolisten tcp \
  >build/nsys-gui/xvfb.log 2>&1 </dev/null &
printf '%s\n' "$!" >build/nsys-gui/xvfb.pid
```

确认下面的命令成功后继续；如果失败，查看 `build/nsys-gui/xvfb.log`。

```bash
DISPLAY=:99 xdpyinfo >/dev/null
DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa glxinfo -B
```

```bash
nohup env DISPLAY=:99 /usr/bin/openbox \
  >build/nsys-gui/openbox.log 2>&1 </dev/null &
printf '%s\n' "$!" >build/nsys-gui/openbox.pid

nohup /usr/bin/x11vnc -display :99 -localhost -rfbport 5901 \
  -forever -shared -nopw -noxdamage \
  >build/nsys-gui/x11vnc.log 2>&1 </dev/null &
printf '%s\n' "$!" >build/nsys-gui/x11vnc.pid

nohup /usr/bin/websockify --web=/usr/share/novnc \
  127.0.0.1:6080 127.0.0.1:5901 \
  >build/nsys-gui/websockify.log 2>&1 </dev/null &
printf '%s\n' "$!" >build/nsys-gui/websockify.pid
```

随后按上一节设置环境变量并启动 `nsys-ui`。如果页面能打开但无法连接桌面，检查 x11vnc 与 websockify 日志；如果无法加载报告，检查 GUI 弹窗指出的动态库。

## 本次缺库问题的原因

`libOpenGL.so.0` 来自 `libopengl0`；进一步启动时还发现 `libEGL.so.1` 和报告插件所需的 `libsmime3.so` 缺失，分别通过 `libegl1` 与 `libnss3` 补齐。`libjpeg62` 补齐 Qt 的 JPEG 插件依赖。

`nsys-ui` 启动脚本会先探测 OpenGL，再决定是否回退到软件渲染。没有显示服务时，探测也会失败，因此报出的 OpenGL 版本 `0` 不能直接用来判断 GPU 的硬件能力。动态库、显示入口和实际渲染能力需要分别检查。

来源：[NVIDIA GUI 故障排查](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gui-troubleshooting)、[noVNC 官方说明](https://github.com/novnc/noVNC)。
