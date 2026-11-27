# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def get_version() -> str:
    with (ROOT / "mesd" / "_version.py").open(encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"__version__\W*=\W*\"([^\"]+)\""
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> list[str]:
    with (ROOT / "requirements.txt").open(encoding="utf-8") as f:
        file_content = f.read()
        lines = [
            line.strip()
            for line in file_content.strip().split("\n")
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return lines


def main():
    setup(
        name="recast-video-training",
        version=get_version(),
        description="RECAST reinforcement learning for video reasoning",
        long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        license="Apache 2.0 License",
        package_dir={"": "."},
        packages=find_packages(where=".", include=["verl", "verl.*", "mesd", "mesd.*"]),
        python_requires=">=3.11.0",
        install_requires=get_requires(),
    )


if __name__ == "__main__":
    main()
