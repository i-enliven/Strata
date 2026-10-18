"""Tests for setup on an arm64 PC with an integrated NVIDIA GPU (the DGX Spark, docs/SPARK.md): a CPU whose
/proc/cpuinfo has no `flags:` line, a GPU whose nvidia-smi reports no separate memory, and the engine that has to be
compiled here because the releases carry no arm64 asset. Runs on any PC: no GPU, no network, nothing installed.

    python -m unittest tools.test_setup_spark
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import urllib.error
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


# nvidia-smi on a GB10: the card is there, its compute capability is there, its memory is not - there is no separate
# VRAM to report.  This is the real line from the box the port is developed on.
GB10_NVIDIA_SMI = "0, NVIDIA GB10, [N/A], 12.1, 595.99.02\n"


class GpuWithoutReportedMemory(unittest.TestCase):
    def test_a_card_that_reports_no_memory_is_still_a_card(self):
        # nvidia-smi on a GB10 answers "[N/A]" for memory.total; float() raised and the row was dropped
        found = setup.parse_gpu_rows(GB10_NVIDIA_SMI)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["arch"], "121")
        self.assertTrue(found[0]["mem_na"])

    def test_a_unified_card_is_measured_from_the_ram_it_can_reach(self):
        g = setup.nvidia_apply_uma({"name": "NVIDIA GB10", "arch": "121", "mem_na": True, "integrated": True,
                                    "vram_gb": 0.0}, ram=121.0)
        self.assertTrue(g["uma"])
        self.assertEqual(g["dedicated_gb"], 0.0)
        self.assertAlmostEqual(g["vram_gb"], 121.0 - setup.UMA_OS_LEFT_GB, places=6)

    def test_a_unified_card_is_named_as_unified_memory(self):
        g = setup.nvidia_apply_uma({"name": "NVIDIA GB10", "arch": "121", "mem_na": True, "integrated": True,
                                    "vram_gb": 0.0}, ram=121.0)
        self.assertIn("unified memory", setup.nvidia_mem_text(g))

    def test_a_unified_card_adds_nothing_beside_the_ram_in_low_ram_mode(self):
        # the rule low_ram_vram() already makes for an AMD APU: its shared memory IS the RAM
        spark = setup.nvidia_apply_uma({"name": "NVIDIA GB10", "arch": "121", "mem_na": True, "integrated": True,
                                        "vram_gb": 0.0}, ram=121.0)
        self.assertEqual(setup.low_ram_vram(spark), 0.0)

    def test_a_card_that_reports_no_memory_but_is_not_integrated_is_not_given_the_ram(self):
        g = setup.nvidia_apply_uma({"name": "Some vGPU", "arch": "89", "mem_na": True, "integrated": False,
                                    "vram_gb": 0.0}, ram=64.0)
        self.assertFalse(g.get("uma"))


class CpuShareOnArm(unittest.TestCase):
    def test_asking_for_the_cpu_share_on_arm64_says_so(self):
        src = (ROOT / "src" / "prefill" / "prefill.cpp").read_text(encoding="utf-8")
        self.assertIn("__aarch64__", src[src.index("void Prefill::arm_cpu_share"):][:900])
        self.assertIn("this build has no CPU expert kernels", src)

    def test_the_share_is_not_armed_on_arm64(self):
        # setting g_stream_min_share here would hand expert rows to a CPU pool whose kernels stop the engine
        src = (ROOT / "src" / "prefill" / "prefill.cpp").read_text(encoding="utf-8")
        body = src[src.index("void Prefill::arm_cpu_share"):src.index("bool Prefill::ring_bytes_enabled")]
        self.assertIn("#if !defined(__aarch64__)", body)


class CMakeArm64Branch(unittest.TestCase):
    def test_the_avx_kernel_files_are_built_only_on_an_x86_host(self):
        cm = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("STRATA_HOST_X86", cm)
        for f in ("src/kernels/cpu/expert.cpp", "src/kernels/cpu/q2_avx2.cpp", "src/kernels/cpu/iq_avx512.cpp"):
            self.assertLess(cm.index("if(STRATA_HOST_X86)"), cm.index(f), f)   # inside the guarded block, not before it

    def test_the_arm64_branch_cannot_be_reached_from_the_x86_branch(self):
        cm = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("CMAKE_SYSTEM_PROCESSOR MATCHES", cm)


class NoPrebuiltEngineOnArm(unittest.TestCase):
    def test_the_arm64_host_does_not_probe_for_a_ready_made_engine(self):
        # the releases carry strata-linux-x64.zip and nothing for arm64; get_prebuilt's HEAD probe only waits
        def no_network(*a, **kw):
            raise AssertionError("setup reached the network on arm64")
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "engine_dir", lambda toolkit=13: Path(d) / "engine"), \
                mock.patch.object(setup, "ARM64", True), \
                mock.patch.object(setup.urllib.request, "urlopen", no_network), \
                contextlib.redirect_stdout(out):
            self.assertIsNone(setup.get_prebuilt(setup.PREBUILT_URL, {"arch": "121"}, "gpu"))
        self.assertIn("compiled here", out.getvalue())

    def test_an_x86_host_still_looks_for_the_ready_made_engine(self):
        # the default path AGENTS.md keeps: the probe runs, and its failure is what falls back to compiling
        seen = []

        def probe(req, *a, **kw):
            seen.append(req.full_url)
            raise urllib.error.HTTPError(req.full_url, 404, "not published", {}, None)

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "engine_dir", lambda toolkit=13: Path(d) / "engine"), \
                mock.patch.object(setup, "ARM64", False), \
                mock.patch.object(setup.urllib.request, "urlopen", probe), \
                contextlib.redirect_stdout(out):
            self.assertIsNone(setup.get_prebuilt(setup.PREBUILT_URL, {"arch": "120"}, "gpu"))
        self.assertTrue(seen, "an x86 host must still probe the release asset")


class CheckMessagesOnArm(unittest.TestCase):
    def test_an_arm_cpu_is_not_reported_as_lacking_avx2(self):
        # "no AVX2" reads like a CPU too old for Strata; on Arm the x86 CPU kernels are simply not built
        with mock.patch.object(setup, "ARM64", True):
            self.assertEqual(setup.cpu_feature_note(False, False), "Arm")

    def test_an_x86_cpu_is_named_by_what_it_has(self):
        with mock.patch.object(setup, "ARM64", False):
            self.assertEqual(setup.cpu_feature_note(False, False), "no AVX2")
            self.assertEqual(setup.cpu_feature_note(True, False), "AVX2")
            self.assertEqual(setup.cpu_feature_note(True, True), "AVX-512")

    def test_a_unified_gpu_has_no_pcie_link_to_measure(self):
        # the GB10 sits on the SoC: the probe read "PCIe: 1.0 x1 ... up to ~0 GB/s" and warned about lanes
        self.assertFalse(setup.pcie_applies({"uma": True}))
        self.assertTrue(setup.pcie_applies({"vram_gb": 24.0}))

    def test_the_older_cpu_warning_is_not_shown_where_there_is_no_x86_floor(self):
        # the warning names SSE4.2 and STRATA_ISA_FLOOR, neither of which means anything on Arm
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("if not avx2 and floor:", src)


class CpuFloorOnArm(unittest.TestCase):
    def test_an_arm_host_has_no_x86_instruction_floor(self):
        # cpu_floor() answers "unsupported" for a CPU without AVX2, and main() stops the install on that (setup.py:6356)
        with mock.patch.object(setup, "ARM64", True), mock.patch.object(setup, "WIN", False):
            self.assertEqual(setup.cpu_floor(False), "")


if __name__ == "__main__":
    unittest.main()