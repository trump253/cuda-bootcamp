#!/usr/bin/env python3
"""准备固定版本的 noVNC，使用简体中文和适合时间线查看的默认设置。"""

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


VERSION = "1.7.0"
SHA256 = "b1003a11b6e6e8d8f7f5e5586daae7f8ca651d8aee0aa155ff9ac841c48f52c6"
URL = f"https://codeload.github.com/novnc/noVNC/tar.gz/refs/tags/v{VERSION}"


def main():
    root = Path(__file__).resolve().parent.parent / "build" / "nsys-gui"
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"noVNC-{VERSION}.tar.gz"
    if archive.exists():
        blob = archive.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=30) as response:
            blob = response.read()
    if hashlib.sha256(blob).hexdigest() != SHA256:
        raise SystemExit("noVNC 归档校验失败，请检查下载来源和文件内容。")
    archive.write_bytes(blob)

    target = root / f"noVNC-{VERSION}"
    target.mkdir(exist_ok=True)
    # 只解压普通文件与目录，并限制到本次生成的资源目录。
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts or ".." in parts:
                continue
            path = target.joinpath(*parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output)

    # 此浏览器入口按学习者偏好使用简体中文，不依赖浏览器的语言顺序。
    ui_path = target / "app" / "ui.js"
    original = 'await l10n.setup(LINGUAS, "app/locale/");'
    replacement = '''l10n.language = "zh_CN";
            await l10n._setupDictionary("app/locale/");
            document.documentElement.lang = "zh-CN";'''
    ui = ui_path.read_text()
    if ui.count(original) != 1:
        raise SystemExit("noVNC 初始化代码与预期不符，请检查固定版本。")
    ui_path.write_text(ui.replace(original, replacement))

    defaults = {
        "autoconnect": True,
        "resize": "scale",
        "quality": 5,
        "compression": 2,
        "reconnect": True,
        "reconnect_delay": 1000,
    }
    (target / "defaults.json").write_text(json.dumps(defaults, indent=2) + "\n")
    (target / "mandatory.json").write_text("{}\n")

    # 为整套资源使用版本路径，避免升级后新网页混用旧缓存的模块。
    web = root / "web"
    web.mkdir(exist_ok=True)
    link = web / target.name
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise SystemExit(f"网页资源入口已被其他文件占用：{link}")
    link.symlink_to(Path("..") / target.name, target_is_directory=True)
    # 桌面入口的相对资源仍从版本目录加载。
    page = (target / "vnc.html").read_text()
    if page.count("<head>") != 1:
        raise SystemExit("noVNC 网页结构与预期不符，请检查固定版本。")
    entry = page.replace("<head>", f'<head>\n    <base href="./{target.name}/">', 1)
    (web / "vnc.html").write_text(entry)
    # 保留服务器的目录首页，由学习者点击 vnc.html 进入桌面。
    (web / "index.html").unlink(missing_ok=True)
    print(f"noVNC {VERSION} 简体中文资源已准备：{target}")
    print(f"websockify 网页根目录：{web}")


if __name__ == "__main__":
    main()
