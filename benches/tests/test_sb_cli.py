"""Tests for sb-cli: init, fork, and collect commands.

Uses pytest tmp_path fixtures — no real synthesis runs or torch-mlir required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_dir(tmp_path: Path) -> Path:
    """Set up a minimal base directory structure for testing."""
    (tmp_path / "experiments").mkdir()
    return tmp_path


def _run_scaffold(
    tmp_path: Path,
    output_dir: str | None = "test_exp",
    benchmark_name: str | None = None,
    dataset: str = "MINI",
    dtype: str = "float32",
    device: str = "nangate45",
    clock_period: float = 5.0,
    memory_policy: str = "",
    flow: str = "baseline",
    backend: str = "bambu",
    stage: str = "verilog",
) -> Path:
    """Run scaffold() with given args and return the experiment dir."""
    from sb_cli.flow import ExperimentConfig
    from sb_cli.init import scaffold

    config = ExperimentConfig(
        benchmark_name=benchmark_name,
        dataset=dataset,
        dtype=dtype,
        device=device,
        clock_period=clock_period,
        memory_policy=memory_policy,
        flow=flow,
        backend=backend,
        stage=stage,
    )
    return scaffold(config, output_dir, tmp_path)


# ---------------------------------------------------------------------------
# Target resolution: flow x backend x stage
# ---------------------------------------------------------------------------


_GDS_TAIL = "HLS_output/Synthesis/bash_flow/openroad/results/nangate45"


class TestResolveTarget:
    @pytest.mark.parametrize("flow", ["baseline", "optimized", "transformed"])
    @pytest.mark.parametrize(
        ("stage", "expected"),
        [
            ("llvm", "$(ODIR)/05_llvm_{flow}.ll"),
            ("verilog", "$(ODIR)/bambu/{flow}/06_verilog.v"),
            ("simulation", "$(ODIR)/bambu/{flow}/07_results.txt"),
            (
                "gds",
                "$(ODIR)/bambu/{flow}/"
                + _GDS_TAIL
                + "/forward_kernel/base/6_final.gds",
            ),
        ],
    )
    def test_every_flow_stage_pair(self, flow: str, stage: str, expected: str) -> None:
        """All 12 (flow, stage) combinations resolve to the mkinc rule paths."""
        from sb_cli.flow import resolve_target

        got = resolve_target(flow, "bambu", stage, device="nangate45")
        assert got == expected.format(flow=flow)

    def test_gds_platform_strips_corner_suffix(self) -> None:
        """Bambu corner suffixes are not part of the OpenROAD results path."""
        from sb_cli.flow import gds_platform

        assert gds_platform("asap7-BC") == "asap7"
        assert gds_platform("nangate45") == "nangate45"

    def test_gds_path_uses_device_platform(self) -> None:
        """The gds target follows --device rather than hardcoding nangate45."""
        from sb_cli.flow import resolve_target

        got = resolve_target("baseline", "bambu", "gds", device="asap7-BC")
        assert "/openroad/results/asap7/" in got

    @pytest.mark.parametrize(
        ("flow", "backend", "stage"),
        [
            ("nonsense", "bambu", "verilog"),
            ("baseline", "nonsense", "verilog"),
            ("baseline", "bambu", "nonsense"),
        ],
    )
    def test_unknown_axis_value_raises(
        self, flow: str, backend: str, stage: str
    ) -> None:
        """An unknown value on any axis is rejected with the legal choices."""
        from sb_cli.flow import resolve_target

        with pytest.raises(ValueError, match="nonsense"):
            resolve_target(flow, backend, stage, device="nangate45")

    def test_removed_target_flag_errors(self, capsys: pytest.CaptureFixture) -> None:
        """The removed --target flag fails loudly and names its replacements."""
        import sys
        from unittest.mock import patch

        from sb_cli.__main__ import main

        argv = ["sb-cli", "init", "--output_dir", "x", "--target", "transformed"]
        with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
            main()

        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert "--flow" in err and "--stage" in err

    def test_unsupported_pair_raises(self) -> None:
        """A legal backend with no template for a legal stage is rejected."""
        from sb_cli import flow as flow_mod

        # Simulate a future backend that only reaches llvm.
        original = flow_mod.BACKENDS
        flow_mod.BACKENDS = (*original, "cpu")
        try:
            with pytest.raises(ValueError, match="does not support stage 'gds'"):
                flow_mod.resolve_target("baseline", "cpu", "gds", device="nangate45")
        finally:
            flow_mod.BACKENDS = original


# ---------------------------------------------------------------------------
# T015: init tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_default_no_benchmark(self, tmp_path: Path) -> None:
        """Scaffold without benchmark_name uses default torchscript template."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="my_exp")

        # All six files present
        for fname in [
            "torchscript.py",
            "flow.py",
            "Makefile",
            "transform.mlir",
            "README.md",
            ".gitignore",
        ]:
            assert (exp_dir / fname).exists(), f"Missing: {fname}"

        # Symlink exists and resolves
        symlink = base / "experiments" / "my_exp"
        assert symlink.is_symlink()
        assert symlink.resolve() == exp_dir.resolve()

        # Registry entry exists
        registry_path = base / "experiments" / "registry.py"
        assert registry_path.exists()
        content = registry_path.read_text()
        assert '"my_exp"' in content

        # torchscript.py uses default MM class (no benchmark import)
        ts_content = (exp_dir / "torchscript.py").read_text()
        assert "class MM" in ts_content
        assert "torch_mlir" in ts_content

    def test_init_generates_correct_makefile(self, tmp_path: Path) -> None:
        """Makefile has correct SCRIPTS_DIR and device defaults."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base, output_dir="mk_exp", device="nangate45", clock_period=5.0
        )

        mk = (exp_dir / "Makefile").read_text()
        assert "SCRIPTS_DIR=../../../scripts" in mk
        assert "BAMBU_DEVICE?=nangate45" in mk
        assert "BAMBU_CLOCK_PERIOD?=5.0" in mk
        assert "tosa_to_llvm.mk" in mk
        assert "soda_to_llvm.mk" in mk
        assert "llvm_to_verilog.mk" in mk
        assert "verilog_to_gds.mk" in mk

    def test_init_target_verilog(self, tmp_path: Path) -> None:
        """TARGET in Makefile matches the verilog target path."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="vlog_exp", stage="verilog")

        mk = (exp_dir / "Makefile").read_text()
        assert "bambu/baseline/06_verilog.v" in mk

        flow = (exp_dir / "flow.py").read_text()
        assert "bambu/baseline/06_verilog.v" in flow

    def test_init_target_gds(self, tmp_path: Path) -> None:
        """TARGET in Makefile matches the gds target path."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="gds_exp", stage="gds")

        mk = (exp_dir / "Makefile").read_text()
        assert "6_final.gds" in mk

    def test_init_gds_honors_flow(self, tmp_path: Path) -> None:
        """The gds TARGET follows --flow instead of being pinned to baseline.

        Regression test: the old TARGET_MAP["gds"] hardcoded bambu/baseline,
        so a transformed experiment silently synthesized the baseline design.
        """
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base, output_dir="gds_xfm", flow="transformed", stage="gds"
        )

        mk = (exp_dir / "Makefile").read_text()
        assert "bambu/transformed/" in mk
        assert "bambu/baseline/" not in mk

    def test_init_stage_llvm_is_flow_aware(self, tmp_path: Path) -> None:
        """--stage llvm resolves to 05_llvm_<flow>.ll, never the tosa 04_llvm.ll."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base, output_dir="llvm_opt", flow="optimized", stage="llvm"
        )

        mk = (exp_dir / "Makefile").read_text()
        assert "TARGET=$(ODIR)/05_llvm_optimized.ll" in mk
        assert "04_llvm.ll" not in mk

    def test_init_stage_simulation(self, tmp_path: Path) -> None:
        """--stage simulation reaches the 07_results.txt rule."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base, output_dir="sim_exp", flow="transformed", stage="simulation"
        )

        mk = (exp_dir / "Makefile").read_text()
        assert "bambu/transformed/07_results.txt" in mk

    def test_init_transform_mlir_is_noop(self, tmp_path: Path) -> None:
        """transform.mlir is generated as a no-op boilerplate."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="xfm_exp")

        xfm = (exp_dir / "transform.mlir").read_text()
        assert "transform.with_named_sequence" in xfm
        assert "transform.yield" in xfm

    def test_init_gitignore_ignores_output(self, tmp_path: Path) -> None:
        """.gitignore inside experiment ignores output/."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="gi_exp")

        gi = (exp_dir / ".gitignore").read_text()
        assert "output/" in gi

    def test_init_timestamped_dir_created(self, tmp_path: Path) -> None:
        """A timestamped directory is created under experiments/."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="ts_exp")

        # The actual experiment dir name is a timestamp
        assert exp_dir.parent == base / "experiments"
        # Timestamp dirs match YYYY_MM_DD_HH_MM_SS pattern
        import re

        assert re.match(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", exp_dir.name)

    def test_init_registry_appended(self, tmp_path: Path) -> None:
        """Multiple init calls append distinct entries to registry."""
        base = _make_base_dir(tmp_path)
        _run_scaffold(base, output_dir="exp_a")
        _run_scaffold(base, output_dir="exp_b")

        from sb_cli.registry import Registry

        reg = Registry(base)
        experiments = reg.load()
        assert "exp_a" in experiments
        assert "exp_b" in experiments
        assert experiments["exp_a"] != experiments["exp_b"]

    def test_init_memory_policy_in_flow(self, tmp_path: Path) -> None:
        """memory_policy is reflected in generated flow.py."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="mp_exp", memory_policy="NO_BRAM")

        flow = (exp_dir / "flow.py").read_text()
        assert "NO_BRAM" in flow

    def test_init_readme_contains_config(self, tmp_path: Path) -> None:
        """README.md contains experiment name and device."""
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base, output_dir="readme_exp", device="asap7-BC", dataset="MEDIUM"
        )

        readme = (exp_dir / "README.md").read_text()
        assert "readme_exp" in readme
        assert "asap7-BC" in readme
        assert "MEDIUM" in readme

    def test_init_with_benchmark_name(self, tmp_path: Path) -> None:
        """Scaffold with benchmark_name imports from the specified module."""
        base = _make_base_dir(tmp_path)
        # Use a real PolyBench module that we know exists
        exp_dir = _run_scaffold(
            base,
            output_dir="gemm_exp",
            benchmark_name="PolyBenchPyTorch.linear_algebra.blas.gemm",
        )

        ts = (exp_dir / "torchscript.py").read_text()
        # Should import from the benchmark module, not define MM class
        assert "PolyBenchPyTorch.linear_algebra.blas.gemm" in ts
        assert "generate_mlir" in ts
        assert "class MM" not in ts

    def test_init_benchmark_name_imports_kernel_submodule(self, tmp_path: Path) -> None:
        """The generated import targets the kernel's implementation submodule.

        Kernel packages (e.g. `...blas.gemm`) have an empty `__init__.py`;
        the `nn.Module` subclass and `init_array` live in the same-named
        submodule (`...blas.gemm.gemm`). The generated torchscript.py must
        import from that submodule, not the package, or it fails at runtime
        with `ImportError: cannot import name 'Gemm'`.
        """
        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base,
            output_dir="gemm_exp",
            benchmark_name="PolyBenchPyTorch.linear_algebra.blas.gemm",
        )

        ts = (exp_dir / "torchscript.py").read_text()
        assert (
            "from benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm "
            "import Gemm, init_array" in ts
        )

    def test_init_accepts_short_benchmark_name(self, tmp_path: Path) -> None:
        """A catalog short name resolves to the same kernel as the long path."""
        base = _make_base_dir(tmp_path)
        short = _run_scaffold(base, output_dir="short_exp", benchmark_name="gemm")
        long = _run_scaffold(
            base,
            output_dir="long_exp",
            benchmark_name="PolyBenchPyTorch.linear_algebra.blas.gemm",
        )

        short_ts = (short / "torchscript.py").read_text()
        assert "import Gemm, init_array" in short_ts
        # Only the docstring header differs (it echoes benchmark_name verbatim)
        assert "from benches.PolyBenchPyTorch" in short_ts
        assert "from benches.PolyBenchPyTorch" in (long / "torchscript.py").read_text()

    def test_generated_torchscript_imports_resolve_from_any_cwd(
        self, tmp_path: Path
    ) -> None:
        """The generated torchscript.py imports must not depend on the cwd.

        It is executed from experiments/<name>/ by the generated Makefile, so a
        kernel path that only resolves inside benches/ fails at runtime with
        ModuleNotFoundError. Runs the file's own import statements in a
        subprocess from an unrelated directory.
        """
        import ast
        import subprocess
        import sys

        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base,
            output_dir="cwd_exp",
            benchmark_name="PolyBenchPyTorch.linear_algebra.blas.gemm",
        )

        source = (exp_dir / "torchscript.py").read_text()
        tree = ast.parse(source)
        imports = "\n".join(
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, ast.Import | ast.ImportFrom)
        )
        assert "benches" in imports, "generated imports are not fully qualified"

        # Inherits the real environment (the repo is installed editable), so a
        # failure here means the imports were cwd-dependent, nothing else.
        result = subprocess.run(
            [sys.executable, "-c", imports],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"generated imports failed from cwd={tmp_path}:\n{result.stderr}"
        )

    def test_init_duplicate_output_dir_exits(self, tmp_path: Path) -> None:
        """Second init with same output_dir raises SystemExit."""
        base = _make_base_dir(tmp_path)
        _run_scaffold(base, output_dir="dup_exp")

        with pytest.raises(SystemExit):
            _run_scaffold(base, output_dir="dup_exp")

    def test_init_bad_benchmark_name_exits(self, tmp_path: Path) -> None:
        """Non-importable benchmark_name raises SystemExit."""
        base = _make_base_dir(tmp_path)
        with pytest.raises(SystemExit):
            _run_scaffold(
                base, output_dir="bad_exp", benchmark_name="no.such.module.Xyz"
            )


# ---------------------------------------------------------------------------
# Auto-naming: --output_dir is optional for init and fork
# ---------------------------------------------------------------------------


class TestInitAutoName:
    def test_init_auto_name_from_benchmark(self, tmp_path: Path) -> None:
        """Omitting output_dir names the experiment <bench>-<dataset>-<dtype>."""
        from sb_cli.registry import Registry

        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir=None, benchmark_name="gemm")

        symlink = base / "experiments" / "gemm-MINI-float32"
        assert symlink.is_symlink()
        assert symlink.resolve() == exp_dir.resolve()
        assert "gemm-MINI-float32" in Registry(base).load()

    def test_init_auto_name_uses_dataset_and_dtype(self, tmp_path: Path) -> None:
        """dataset and dtype are part of the auto-generated name."""
        base = _make_base_dir(tmp_path)
        _run_scaffold(
            base,
            output_dir=None,
            benchmark_name="gemm",
            dataset="MEDIUM",
            dtype="float16",
        )

        assert (base / "experiments" / "gemm-MEDIUM-float16").is_symlink()

    def test_init_auto_name_collision_appends_suffix(self, tmp_path: Path) -> None:
        """Repeated auto-named inits fall back to an incrementing -NNN counter."""
        from sb_cli.registry import Registry

        base = _make_base_dir(tmp_path)
        dirs = [
            _run_scaffold(base, output_dir=None, benchmark_name="gemm")
            for _ in range(3)
        ]

        names = ["gemm-MINI-float32", "gemm-MINI-float32-000", "gemm-MINI-float32-001"]
        experiments = Registry(base).load()
        for name, exp_dir in zip(names, dirs, strict=True):
            symlink = base / "experiments" / name
            assert symlink.is_symlink(), f"Missing symlink: {name}"
            assert symlink.resolve() == exp_dir.resolve()
            assert name in experiments
        # Each run got its own timestamped directory
        assert len({d.name for d in dirs}) == 3

    def test_init_auto_name_requires_benchmark(self, tmp_path: Path) -> None:
        """Without benchmark_name there is nothing to derive a name from."""
        base = _make_base_dir(tmp_path)
        with pytest.raises(SystemExit):
            _run_scaffold(base, output_dir=None, benchmark_name=None)

    def test_next_available_name_fills_gap(self, tmp_path: Path) -> None:
        """next_available_name returns the first free counter, not the last + 1."""
        from sb_cli.init import next_available_name

        base = _make_base_dir(tmp_path)
        (base / "experiments" / "foo-000").mkdir()
        (base / "experiments" / "foo-002").mkdir()

        assert next_available_name("foo", base) == "foo-001"

    def test_next_available_name_strips_existing_counter(self, tmp_path: Path) -> None:
        """A stem that already ends in -NNN continues the series, not nests."""
        from sb_cli.init import next_available_name

        base = _make_base_dir(tmp_path)
        (base / "experiments" / "foo-000").mkdir()

        assert next_available_name("foo-000", base) == "foo-001"

    def test_next_available_name_skips_registry_only_entry(
        self, tmp_path: Path
    ) -> None:
        """A registered name with no directory on disk still counts as taken."""
        from sb_cli.init import next_available_name
        from sb_cli.registry import Registry

        base = _make_base_dir(tmp_path)
        Registry(base).append("foo-000", "experiments/2026_01_01_00_00_00")

        assert next_available_name("foo", base) == "foo-001"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_created_on_first_append(self, tmp_path: Path) -> None:
        """Registry file is created automatically on first append."""
        from sb_cli.registry import Registry

        (tmp_path / "experiments").mkdir()
        reg = Registry(tmp_path)
        assert not (tmp_path / "experiments" / "registry.py").exists()

        reg.append("foo", "experiments/2026_01_01_00_00_00")
        assert (tmp_path / "experiments" / "registry.py").exists()

    def test_registry_load_returns_dict(self, tmp_path: Path) -> None:
        """load() returns a dict after appending entries."""
        from sb_cli.registry import Registry

        (tmp_path / "experiments").mkdir()
        reg = Registry(tmp_path)
        reg.append("a", "experiments/ts_a")
        reg.append("b", "experiments/ts_b")

        data = reg.load()
        assert data["a"] == "experiments/ts_a"
        assert data["b"] == "experiments/ts_b"

    def test_registry_resolve_by_name(self, tmp_path: Path) -> None:
        """resolve() finds an experiment by registered name."""
        from sb_cli.registry import Registry

        (tmp_path / "experiments").mkdir()
        ts_dir = tmp_path / "experiments" / "2026_01_01_00_00_00"
        ts_dir.mkdir()

        reg = Registry(tmp_path)
        reg.append("myexp", "experiments/2026_01_01_00_00_00")

        resolved = reg.resolve("myexp", tmp_path)
        assert resolved == ts_dir.resolve()

    def test_registry_resolve_by_path(self, tmp_path: Path) -> None:
        """resolve() falls back to direct path when name not in registry."""
        from sb_cli.registry import Registry

        (tmp_path / "experiments").mkdir()
        ts_dir = tmp_path / "experiments" / "2026_01_01_00_00_00"
        ts_dir.mkdir()

        reg = Registry(tmp_path)
        resolved = reg.resolve(str(ts_dir), tmp_path)
        assert resolved == ts_dir.resolve()

    def test_registry_resolve_missing_exits(self, tmp_path: Path) -> None:
        """resolve() raises SystemExit when name not in registry and path missing."""
        from sb_cli.registry import Registry

        (tmp_path / "experiments").mkdir()
        reg = Registry(tmp_path)

        with pytest.raises(SystemExit):
            reg.resolve("nonexistent", tmp_path)


# ---------------------------------------------------------------------------
# T018: flow.py template test — generated flow.py sets correct env vars
# ---------------------------------------------------------------------------


class TestGeneratedFlow:
    def test_flow_run_sets_env_vars(self, tmp_path: Path) -> None:
        """Generated flow.py invokes make with correct BAMBU_* env vars."""
        import importlib.util
        import subprocess
        from unittest.mock import patch

        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base,
            output_dir="flow_test",
            device="asap7-BC",
            clock_period=3.5,
            memory_policy="NO_BRAM",
            stage="verilog",
        )

        # Dynamically import the generated flow.py
        spec = importlib.util.spec_from_file_location("gen_flow", exp_dir / "flow.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        captured_env: dict = {}

        def fake_run(cmd, env, check, cwd):  # noqa: ANN001
            captured_env.update(env)

        with patch.object(subprocess, "run", side_effect=fake_run):
            mod.Flow().run()

        assert captured_env["BAMBU_DEVICE"] == "asap7-BC"
        assert captured_env["BAMBU_CLOCK_PERIOD"] == "3.5"
        assert captured_env["BAMBU_MEMPOLICY"] == "NO_BRAM"

    def test_flow_run_no_memory_policy(self, tmp_path: Path) -> None:
        """Generated flow.py omits BAMBU_MEMPOLICY when memory_policy is empty."""
        import importlib.util
        import subprocess
        from unittest.mock import patch

        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(
            base,
            output_dir="flow_nopol",
            memory_policy="",
        )

        spec = importlib.util.spec_from_file_location("gen_flow2", exp_dir / "flow.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        captured_env: dict = {}

        def fake_run(cmd, env, check, cwd):  # noqa: ANN001
            captured_env.update(env)

        with patch.object(subprocess, "run", side_effect=fake_run):
            mod.Flow().run()

        policy = captured_env.get("BAMBU_MEMPOLICY", "")
        assert policy == ""


# ---------------------------------------------------------------------------
# T022: fork tests
# ---------------------------------------------------------------------------


class TestFork:
    def test_fork_copies_files(self, tmp_path: Path) -> None:
        """Fork hard-copies all six tracked files into new directory."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        _run_scaffold(base, output_dir="source_exp")

        fork_experiment("source_exp", "forked_exp", base)

        symlink = base / "experiments" / "forked_exp"
        assert symlink.is_symlink()
        fork_dir = symlink.resolve()

        for fname in [
            "torchscript.py",
            "flow.py",
            "Makefile",
            "transform.mlir",
            "README.md",
            ".gitignore",
        ]:
            assert (fork_dir / fname).exists(), f"Missing in fork: {fname}"

    def test_fork_no_output_dir(self, tmp_path: Path) -> None:
        """Forked experiment has no output/ directory."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        _run_scaffold(base, output_dir="src_exp2")

        fork_dir = fork_experiment("src_exp2", "fork_exp2", base)
        assert not (fork_dir / "output").exists()

    def test_fork_source_unchanged(self, tmp_path: Path) -> None:
        """Editing transform.mlir in the fork does not affect the source."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        src_dir = _run_scaffold(base, output_dir="src_exp3")
        fork_dir = fork_experiment("src_exp3", "fork_exp3", base)

        original = (src_dir / "transform.mlir").read_text()
        (fork_dir / "transform.mlir").write_text("// EDITED\n", encoding="utf-8")

        assert (src_dir / "transform.mlir").read_text() == original
        assert (fork_dir / "transform.mlir").read_text() == "// EDITED\n"

    def test_fork_from_path(self, tmp_path: Path) -> None:
        """fork_experiment accepts a direct filesystem path as source."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        src_dir = _run_scaffold(base, output_dir="path_src")

        # Use direct path (not registry name)
        fork_dir = fork_experiment(str(src_dir), "path_fork", base)
        assert (fork_dir / "Makefile").exists()

    def test_fork_registered(self, tmp_path: Path) -> None:
        """Forked experiment is registered in registry.py."""
        from sb_cli.fork import fork_experiment
        from sb_cli.registry import Registry

        base = _make_base_dir(tmp_path)
        _run_scaffold(base, output_dir="reg_src")
        fork_experiment("reg_src", "reg_fork", base)

        experiments = Registry(base).load()
        assert "reg_src" in experiments
        assert "reg_fork" in experiments

    def test_fork_unresolved_source_exits(self, tmp_path: Path) -> None:
        """fork_experiment raises SystemExit when source is not found."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        with pytest.raises(SystemExit):
            fork_experiment("no_such_name", "irrelevant", base)


class TestForkAutoName:
    def test_fork_auto_name_series(self, tmp_path: Path) -> None:
        """Repeated forks of one source produce -000, -001, ..."""
        from sb_cli.fork import fork_experiment
        from sb_cli.registry import Registry

        base = _make_base_dir(tmp_path)
        _run_scaffold(base, output_dir=None, benchmark_name="gemm")

        first = fork_experiment("gemm-MINI-float32", None, base)
        second = fork_experiment("gemm-MINI-float32", None, base)

        experiments = Registry(base).load()
        for name, fork_dir in [
            ("gemm-MINI-float32-000", first),
            ("gemm-MINI-float32-001", second),
        ]:
            symlink = base / "experiments" / name
            assert symlink.is_symlink(), f"Missing symlink: {name}"
            assert symlink.resolve() == fork_dir.resolve()
            assert name in experiments
            assert (fork_dir / "Makefile").exists()

    def test_fork_auto_name_continues_series(self, tmp_path: Path) -> None:
        """Forking a fork continues the flat series instead of nesting -000-000."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        _run_scaffold(base, output_dir=None, benchmark_name="gemm")
        fork_experiment("gemm-MINI-float32", None, base)

        fork_experiment("gemm-MINI-float32-000", None, base)

        assert (base / "experiments" / "gemm-MINI-float32-001").is_symlink()
        assert not (base / "experiments" / "gemm-MINI-float32-000-000").exists()

    def test_fork_auto_name_from_path(self, tmp_path: Path) -> None:
        """A path source falls back to the source directory's name as the stem."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        src_dir = _run_scaffold(base, output_dir="path_src")

        fork_dir = fork_experiment(str(src_dir), None, base)

        symlink = base / "experiments" / f"{src_dir.name}-000"
        assert symlink.is_symlink()
        assert symlink.resolve() == fork_dir.resolve()
        assert (fork_dir / "Makefile").exists()

    def test_fork_auto_name_unresolved_source_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        """A missing source exits before any directory is created."""
        from sb_cli.fork import fork_experiment

        base = _make_base_dir(tmp_path)
        with pytest.raises(SystemExit):
            fork_experiment("no_such_name", None, base)

        # Resolving reads registry.py (and byte-compiles it); no experiment
        # directory or symlink may have been left behind.
        leftovers = {
            p.name
            for p in (base / "experiments").iterdir()
            if p.name not in {"registry.py", "__pycache__"}
        }
        assert leftovers == set()


# ---------------------------------------------------------------------------
# T029: collect tests
# ---------------------------------------------------------------------------


class TestCollect:
    def _make_experiment_with_output(self, base: Path, name: str) -> Path:
        """Create a scaffolded experiment with synthetic output files."""
        exp_dir = _run_scaffold(base, output_dir=name)
        out = exp_dir / "output"
        out.mkdir()
        # Synthetic .ll file
        ll_file = out / "05_llvm_baseline.ll"
        ll_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
        # Synthetic .v file
        v_dir = out / "bambu" / "baseline"
        v_dir.mkdir(parents=True)
        v_file = v_dir / "06_verilog.v"
        v_file.write_text("wire a;\nwire b;\n", encoding="utf-8")
        return exp_dir

    def test_collect_single(self, tmp_path: Path) -> None:
        """collect_command writes metrics.json for a single experiment."""
        import json as _json

        from sb_cli.collect import collect_command

        base = _make_base_dir(tmp_path)
        exp_dir = self._make_experiment_with_output(base, "coll_exp")

        collect_command("coll_exp", base)

        metrics_path = exp_dir / "output" / "metrics.json"
        assert metrics_path.exists()
        data = _json.loads(metrics_path.read_text())
        assert "metrics" in data
        assert "ll_line_count" in data["metrics"]
        assert "v_line_count" in data["metrics"]
        assert "file_inventory" in data["metrics"]

        # Verify line counts
        ll_results = data["metrics"]["ll_line_count"]
        assert len(ll_results) == 1
        ll_count = next(iter(ll_results.values()))
        assert ll_count == 3

    def test_collect_correct_v_count(self, tmp_path: Path) -> None:
        """v_line_count collector counts lines in .v files."""
        import json as _json

        from sb_cli.collect import collect_command

        base = _make_base_dir(tmp_path)
        exp_dir = self._make_experiment_with_output(base, "v_count_exp")

        collect_command("v_count_exp", base)

        data = _json.loads((exp_dir / "output" / "metrics.json").read_text())
        v_results = data["metrics"]["v_line_count"]
        assert len(v_results) == 1
        assert next(iter(v_results.values())) == 2

    def test_collect_all(self, tmp_path: Path) -> None:
        """collect_command with no arg collects from all registered experiments."""
        import json as _json

        from sb_cli.collect import collect_command

        base = _make_base_dir(tmp_path)
        exp_a = self._make_experiment_with_output(base, "all_a")
        exp_b = self._make_experiment_with_output(base, "all_b")

        collect_command(None, base)

        for exp_dir in [exp_a, exp_b]:
            metrics_path = exp_dir / "output" / "metrics.json"
            assert metrics_path.exists()
            data = _json.loads(metrics_path.read_text())
            assert "metrics" in data

    def test_collect_empty_output(self, tmp_path: Path) -> None:
        """collect_command handles experiment with no output files gracefully."""
        import json as _json

        from sb_cli.collect import collect_command

        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="empty_exp")
        # No output/ directory created

        collect_command("empty_exp", base)

        metrics_path = exp_dir / "output" / "metrics.json"
        assert metrics_path.exists()
        data = _json.loads(metrics_path.read_text())
        # All collectors return empty results
        for key in ["ll_line_count", "v_line_count", "file_inventory"]:
            assert data["metrics"][key] == {}

    def test_collect_from_path(self, tmp_path: Path) -> None:
        """collect_command accepts a direct filesystem path."""
        import json as _json

        from sb_cli.collect import collect_command

        base = _make_base_dir(tmp_path)
        exp_dir = self._make_experiment_with_output(base, "path_coll")

        collect_command(str(exp_dir), base)

        metrics_path = exp_dir / "output" / "metrics.json"
        assert metrics_path.exists()
        data = _json.loads(metrics_path.read_text())
        assert data["experiment"] == exp_dir.name

    def test_collect_metrics_json_structure(self, tmp_path: Path) -> None:
        """metrics.json has collected_at, experiment, and metrics keys."""
        import json as _json

        from sb_cli.collect import collect_command

        base = _make_base_dir(tmp_path)
        exp_dir = _run_scaffold(base, output_dir="struct_exp")

        collect_command("struct_exp", base)

        data = _json.loads((exp_dir / "output" / "metrics.json").read_text())
        assert "collected_at" in data
        assert "experiment" in data
        assert "metrics" in data
        assert data["experiment"] == exp_dir.name
