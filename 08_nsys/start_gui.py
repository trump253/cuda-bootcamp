#!/usr/bin/env python3
"""按就绪状态启动容器内的 Nsight GUI、虚拟显示与浏览器连接服务。"""

import argparse
import datetime
import fcntl
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import configure_novnc


ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "build" / "nsys-gui"
WEB = STATE / "web"


def x_output(display, *args):
    try:
        return subprocess.check_output(
            [args[0], "-display", display, *args[1:]],
            stderr=subprocess.DEVNULL, timeout=2, text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return ""


def desktop_ready(display):
    output = x_output(display, "xprop", "-root", "_NET_SUPPORTING_WM_CHECK")
    match = re.search(r"window id # (0x[0-9a-fA-F]+)", output)
    if not match or int(match[1], 16) == 0:
        return False
    # 验证窗口仍存在，避免只凭根窗口上残留的属性判断。
    return match[1] in x_output(display, "xprop", "-id", match[1], "_NET_SUPPORTING_WM_CHECK")


def vnc_ready(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
            return sock.recv(12).startswith(b"RFB ")
    except OSError:
        return False


def web_ready(port):
    try:
        # 本机服务检查直连，避免被终端中的 HTTP 代理转发。
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/vnc.html", timeout=2) as response:
            return response.status == 200 and b'id="noVNC_container"' in response.read()
    except (OSError, urllib.error.URLError):
        return False


def port_busy(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def launch(name, command, env, state, record_pid=True):
    with (state / f"{name}.log").open("ab") as log:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        log.write(f"\n--- 启动 {now.isoformat(timespec='seconds')} ---\n".encode())
        log.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    if record_pid:
        (state / f"{name}.pid").write_text(f"{process.pid}\n")
    return process


def ensure(name, command, ready, env, state, port=None):
    if ready():
        print(f"复用 {name}，检查通过。", flush=True)
        return None
    if port is not None and port_busy(port):
        raise RuntimeError(f"端口 {port} 已占用但服务检查失败，请检查 {state / (name + '.log')}")
    process = launch(name, command, env, state)
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if ready():
            print(f"启动 {name}，PID {process.pid}，检查通过。", flush=True)
            return process
        time.sleep(0.2)
    raise RuntimeError(f"{name} 未就绪，请检查 {state / (name + '.log')}")


def start_backend(state=STATE, web=WEB, display=":99", vnc_port=5901, web_port=6080):
    """补齐缺失的显示与连接服务；返回本次启动的进程，便于独立验证。"""
    env = os.environ.copy()
    env.update(DISPLAY=display, LIBGL_ALWAYS_SOFTWARE="1",
               __GLX_VENDOR_LIBRARY_NAME="mesa", QTWEBENGINE_CHROMIUM_FLAGS="--no-sandbox")
    state.mkdir(parents=True, exist_ok=True)
    services = [
        ("xvfb", ["Xvfb", display, "-screen", "0", "1600x900x24", "-nolisten", "tcp"],
         lambda: bool(x_output(display, "xdpyinfo")), None),
        ("openbox", ["openbox"], lambda: desktop_ready(display), None),
        ("x11vnc", ["x11vnc", "-display", display, "-localhost", "-rfbport", str(vnc_port),
                    "-forever", "-shared", "-nopw", "-xdamage", "-wait", "5", "-defer", "5", "-sb", "0"],
         lambda: vnc_ready(vnc_port), vnc_port),
        ("websockify", ["websockify", f"--web={web}", f"127.0.0.1:{web_port}", f"127.0.0.1:{vnc_port}"],
         lambda: web_ready(web_port), web_port),
    ]
    started = []
    for name, command, ready, port in services:
        process = ensure(name, command, ready, env, state, port)
        if process is not None:
            started.append(process)
    return env, started


def gui_processes(display):
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            args = (item / "cmdline").read_bytes().split(b"\0")
            if Path(os.fsdecode(args[0])).name != "nsys-ui.bin":
                continue
            environment = (item / "environ").read_bytes().split(b"\0")
            if f"DISPLAY={display}".encode() in environment:
                yield int(item.name), [os.fsdecode(arg) for arg in args if arg]
        except (OSError, ValueError):
            continue


def gui_window_ready(display):
    output = x_output(display, "xprop", "-root", "_NET_CLIENT_LIST")
    for window in re.findall(r"0x[0-9a-fA-F]+", output):
        if "nsys-ui.bin" in x_output(display, "xprop", "-id", window, "WM_CLASS"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, help="要打开的报告，可使用相对路径或绝对路径")
    args = parser.parse_args()
    report = args.report.expanduser().resolve() if args.report is not None else None
    if report is not None and not report.is_file():
        parser.error(f"报告文件不存在：{report}")
    required = ["Xvfb", "openbox", "x11vnc", "websockify", "xdpyinfo", "xprop", "nsys-ui"]
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        parser.error("缺少命令：" + "、".join(missing) + "；依赖安装步骤见 08_nsys/GUI.md。")
    STATE.mkdir(parents=True, exist_ok=True)
    # 同时运行两次启动命令时，后一次等待前一次完成，避免重复绑定端口。
    with (STATE / "start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        resources = [WEB / "vnc.html", WEB / f"noVNC-{configure_novnc.VERSION}" / "app/ui.js",
                     WEB / f"noVNC-{configure_novnc.VERSION}" / "app/locale/zh_CN.json"]
        if not all(path.is_file() for path in resources) or (WEB / "index.html").exists():
            configure_novnc.main()
        env, _ = start_backend()
        existing = list(gui_processes(":99"))
        if existing and report is None:
            print("复用正在运行的 Nsight GUI。", flush=True)
        else:
            if report is None:
                default = ROOT / "build/day10_vector_add.nsys-rep"
                report = default if default.is_file() else None
            command = [shutil.which("nsys-ui")]
            if report is not None:
                command.append(str(report))
            process = launch("nsys-ui", command, env, STATE, record_pid=not existing)
            if existing:
                print("已向 Nsight GUI 发送打开报告请求。", flush=True)
            else:
                print(f"启动 Nsight GUI，PID {process.pid}。", flush=True)
        deadline = time.monotonic() + 15
        while not gui_window_ready(":99"):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"GUI 窗口尚未就绪，请检查 {STATE / 'nsys-ui.log'}")
            time.sleep(0.2)
    print("启动完成。转发远程端口 6080，打开 http://localhost:6080/ 后点击 vnc.html。")
    print(f"日志目录：{STATE}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"启动失败：{error}", file=sys.stderr)
        sys.exit(1)
