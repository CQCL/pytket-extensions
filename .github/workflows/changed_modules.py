import subprocess
import re
import os
import sys


def get_changed_modules(ref_before: str, ref_now: str) -> list:

    process = subprocess.Popen(
        [
            "git",
            "diff",
            "--no-commit-id",
            "--name-only",
            "-r",
            # "25653161e82e41b5b6d8839f6031568a12b1717d",
            # "HEAD"
            ref_before,
            ref_now,
        ],
        stdout=subprocess.PIPE,
    )
    output = str(process.communicate()[0], "utf-8")

    pat = re.compile(r"modules\/([\w-]+)\/.*")

    found_mods = [
        m for m in set(pat.findall(output)) if os.path.isdir("modules/{}".format(m))
    ]

    return found_mods


if __name__ == "__main__":
    print(" ".join(get_changed_modules(sys.argv[1], sys.argv[2])))
