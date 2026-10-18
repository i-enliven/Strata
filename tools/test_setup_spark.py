"""Tests for setup on an arm64 PC with an integrated NVIDIA GPU (the DGX Spark, docs/SPARK.md): a CPU whose
/proc/cpuinfo has no `flags:` line, a GPU whose nvidia-smi reports no separate memory, and the engine that has to be
compiled here because the releases carry no arm64 asset. Runs on any PC: no GPU, no network, nothing installed.

    python -m unittest tools.test_setup_spark
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# /proc/cpuinfo as an Arm board writes it: the feature list is "Features", there is no "flags" line, and the CPU is
# named by "Model" (this is the GB10 box the port is developed on, trimmed to the lines setup reads)
ARM_CPUINFO = """processor\t: 0
BogoMIPS\t: 2000.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc
CPU implementer\t: 0x41
CPU architecture: 8
CPU part\t: 0xd87

processor\t: 1
BogoMIPS\t: 2000.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc
CPU implementer\t: 0x41
CPU architecture: 8
CPU part\t: 0xd87

Model\t: NVIDIA GB10
"""

# an x86 CPU whose cpuinfo carries no flags line at all (a VM that strips it, a patched /proc): setup must not
# promise an AVX-512 engine for a CPU it could not read
X86_WITHOUT_FLAGS = """processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model name\t: Intel(R) Xeon(R) CPU E5-2687W v2 @ 3.40GHz
stepping\t: 4
"""


class CpuWithoutFlagsLine(unittest.TestCase):
    def test_a_cpu_without_a_flags_line_is_read_without_crashing(self):
        # arm64 /proc/cpuinfo spells the line "Features"; setup.py:883 read None.group(1) and died
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", mock.mock_open(read_data=ARM_CPUINFO)):
            name, avx2, avx512 = setup.cpu_info()
        self.assertFalse(avx2)
        self.assertFalse(avx512)
        self.assertNotEqual(name, "")

    def test_an_x86_cpu_without_a_flags_line_says_the_flags_could_not_be_read(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(setup, "WIN", False), \
                mock.patch("builtins.open", mock.mock_open(read_data=X86_WITHOUT_FLAGS)):
            setup.cpu_info()
        self.assertIn("could not read this CPU's flags", out.getvalue())


class CpuFloorOnArm(unittest.TestCase):
    def test_an_arm_host_has_no_x86_instruction_floor(self):
        # cpu_floor() answers "unsupported" for a CPU without AVX2, and main() stops the install on that (setup.py:6356)
        with mock.patch.object(setup, "ARM64", True), mock.patch.object(setup, "WIN", False):
            self.assertEqual(setup.cpu_floor(False), "")


if __name__ == "__main__":
    unittest.main()