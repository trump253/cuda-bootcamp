# 容器内使用 Nsight Systems GUI

本指南针对当前 Ubuntu 20.04 容器、Nsight Systems 2022.4.2，以及 SSH / VS Code Remote SSH 连接方式。2026-10-08 已补齐动态库并启动 Xvfb、Openbox、x11vnc 和 noVNC；随后将 noVNC 升级到 1.7.0，配置简体中文并缩短 VNC 更新等待时间。当前会话先从下面的端口转发开始。

## 在本地看到界面

使用 VS Code Remote SSH 时，打开“端口 / Ports”面板，选择“转发端口”，输入远程端口 `6080`。在本地浏览器打开转发后的根地址。如果 VS Code 分配的本地端口也是 `6080`，使用下面的地址先进入目录首页，再点击 `vnc.html` 打开 noVNC 桌面：

<http://127.0.0.1:6080/>

如果本地端口不同，替换地址中的 `6080`。首页保留服务器自动生成的目录列表；`/vnc.html` 显示桌面，并从 `/noVNC-1.7.0/` 加载资源，避免混用旧版模块。两个入口都没有强制跳转，旧浏览器缓存可用 `Ctrl+F5` 强制刷新。

需要直接进入桌面并显式覆盖浏览器保存的连接与画质设置时，可使用：<http://127.0.0.1:6080/vnc.html?autoconnect=true&resize=scale&quality=5&compression=2>。原来的版本路径入口也可继续使用。

使用普通 SSH 时，在**本地电脑的终端**运行下面的命令，把 `your-container` 换成平时连接这个容器的 SSH 别名或目标；需要特殊 SSH 端口或跳板机时沿用原来的参数。

```bash
ssh -N -L 6080:127.0.0.1:6080 your-container
```

保持隧道运行，在本地浏览器打开上面的入口。转发目标应是当前容器的网络环境。noVNC 与 VNC 只监听容器的回环地址，通过 SSH 访问。

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

## 移除误建的项目

本机 Nsight Systems 2022.4.2 的操作顺序是：在 Project Explorer 中右键误建的项目，先选择 `Unload`；再次右键该项目，选择 `Remove Project`。卸载后才会出现移除选项。本次已按此顺序移除 `Project 1`，当前列表只保留从文件打开的 Vector Add 报告。

`Remove Project` 会将项目移出列表，磁盘目录仍会保留。本次误建的 `/root/.nsightsystems/Projects/Project 1/` 仅包含 112 字节的空项目配置，已移入用户回收站。查看已有 `.nsys-rep` 时，用 `File → Open` 直接打开即可，无需创建采集项目。

## 关闭窗口后重新打开

只关闭本地浏览器标签页时，重新访问上面的 noVNC 入口即可，服务器上的程序继续运行。点击 Nsight Systems 标题栏的关闭按钮后，GUI 程序会退出；此时 noVNC 可能仍然连接着，但只显示黑色桌面。需要在容器的 SSH / VS Code 终端重新启动 GUI：

```bash
cd /root/ai-infra-learning/cuda-bootcamp
mkdir -p build/nsys-gui
nohup env DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa \
  QTWEBENGINE_CHROMIUM_FLAGS=--no-sandbox \
  nsys-ui "$PWD/build/day10_vector_add.nsys-rep" \
  >>build/nsys-gui/nsys-ui.log 2>&1 </dev/null &
printf '%s\n' "$!" >build/nsys-gui/nsys-ui.pid
```

命令会在后台打开 Vector Add 报告，输出追加到 `build/nsys-gui/nsys-ui.log`，浏览器中的窗口会重新出现。查看其他报告时，替换 `.nsys-rep` 路径。已有的 Xvfb、x11vnc 和 websockify 继续使用；整个容器重启后则按后面的完整启动步骤恢复。

## 简体中文与画面流畅度

Ubuntu 自带的 noVNC 1.0 中文翻译使用繁体。本次使用官方 noVNC 1.7.0 的 `zh_CN` 翻译，并固定这个浏览器入口使用简体中文；侧边栏的“连接”“设置”“剪贴板”等已验证。Nsight Systems 程序本身的菜单仍为英文。

升级初次刷新时，学习者遇到 `addTouchSpecificHandlers` 中对 `null` 调用 `addEventListener` 的错误，同时侧边栏仍为繁体；堆栈对应旧版脚本与新版网页混用。已改成上述版本路径，遇到这个现象时在新标签页直接打开新入口。

当前画质为 `5`、压缩等级为 `2`，两者的范围都是 `0–9`。网络较慢时可以在侧边栏“设置”中适当降低画质，减少传输量；字迹变模糊时提高画质。压缩等级越高通常越省带宽，也会增加服务端编码负担。URL 中的参数优先于保存的设置，重新用上述入口打开时会恢复这组参数。

x11vnc 已启用 X DAMAGE，更新等待设置为 `-wait 5 -defer 5`，并使用 `-sb 0` 关闭长时间静止后的慢速轮询。当前显示服务支持 DAMAGE；这些设置会增加轮询频率。本机通过 WebSocket/RFB 测量桌面小区域更新，优化前后各 20 次的延迟中位数约为 `74 ms → 13 ms`。此结果衡量容器内画面更新，不是本地浏览器的 FPS；实际拖动、缩放还受 SSH 网络、画面变化量和 GUI 软件渲染影响。

## `build/nsys-gui/` 的清理与重建

这个目录可以重建，但当前运行 noVNC 时应保留。它被 Git 忽略，仅表示不提交生成文件，并不表示运行时不依赖它。

| 内容 | 用途 |
| --- | --- |
| `noVNC-1.7.0/`、`web/` | 浏览器需要加载的网页、JavaScript、简体中文翻译和配置；websockify 的工作目录是 `web/` |
| `noVNC-1.7.0.tar.gz` | 下载缓存，解压完成后可删除；下次准备资源时需要重新下载 |
| `*.log`、`*.pid` | 运行日志与进程编号，便于排查和管理服务 |
| `screen.png` | 本次验证生成的截图，可删除 |

只关闭 Nsight 程序窗口时，保留目录，使用前面的 GUI 重开命令即可。`mkdir -p` 只能建立空目录，`nsys-ui` 只启动 GUI，二者不会恢复 noVNC 网页资源或启动显示与连接服务。

如果要完整清理，先停止正在运行的 GUI、Openbox、Xvfb、x11vnc 和 websockify，再删除目录。下次恢复时，先运行 `python3 08_nsys/configure_novnc.py` 重建网页资源，再按照下一节恢复显示与连接服务，最后启动 `nsys-ui`。当前 websockify 启动时会切换到 `web/`；若已在运行中删掉目录，即使重新建立同名路径，也需要重启 websockify。

当前报告 `build/day10_vector_add.nsys-rep` 位于这个子目录之外；只删除 `build/nsys-gui/` 不会删除这份报告。

## 容器重启后重新启动

当前组件已经运行。以下步骤用于显示服务退出后重建；重新创建容器时还需要再次安装依赖。命令从仓库根目录执行。

```bash
apt-get install -y --no-install-recommends \
  libopengl0 libegl1 libegl-mesa0 libnss3 libnspr4 libjpeg62 \
  xvfb mesa-utils x11-utils xauth openbox x11vnc websockify python3
```

准备官方固定版本的 noVNC 网页资源。脚本验证下载归档的 SHA-256，设置简体中文和默认画质；重复运行会重新生成相同配置。需要容器能访问 GitHub 下载地址，已下载的归档会复用。

```bash
python3 08_nsys/configure_novnc.py
```

启动虚拟显示并确认它已经就绪，再启动桌面与浏览器转发。网页资源、日志和 PID 写到被 Git 忽略的 `build/nsys-gui/`。

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
  -forever -shared -nopw -xdamage -wait 5 -defer 5 -sb 0 \
  >build/nsys-gui/x11vnc.log 2>&1 </dev/null &
printf '%s\n' "$!" >build/nsys-gui/x11vnc.pid

nohup /usr/bin/websockify --verbose --web="$PWD/build/nsys-gui/web" \
  127.0.0.1:6080 127.0.0.1:5901 \
  >build/nsys-gui/websockify.log 2>&1 </dev/null &
printf '%s\n' "$!" >build/nsys-gui/websockify.pid
```

随后按上一节设置环境变量并启动 `nsys-ui`。如果页面能打开但无法连接桌面，检查 x11vnc 与 websockify 日志；如果无法加载报告，检查 GUI 弹窗指出的动态库。

## 本次缺库问题的原因

`libOpenGL.so.0` 来自 `libopengl0`；进一步启动时还发现 `libEGL.so.1` 和报告插件所需的 `libsmime3.so` 缺失，分别通过 `libegl1` 与 `libnss3` 补齐。`libjpeg62` 补齐 Qt 的 JPEG 插件依赖。

`nsys-ui` 启动脚本会先探测 OpenGL，再决定是否回退到软件渲染。没有显示服务时，探测也会失败，因此报出的 OpenGL 版本 `0` 不能直接用来判断 GPU 的硬件能力。动态库、显示入口和实际渲染能力需要分别检查。

来源：[NVIDIA GUI 故障排查](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gui-troubleshooting)、[noVNC 1.7.0 官方发布](https://github.com/novnc/noVNC/releases/tag/v1.7.0)、[noVNC 画质与配置说明](https://novnc.com/noVNC/docs/EMBEDDING.html)。
