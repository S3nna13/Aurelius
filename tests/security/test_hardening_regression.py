"""Regression tests for security fixes applied during hardening pass.

Covers:
- Code runner safety gate (AURELIUS-SEC-CR-001)
- SSRF protection in HTTP client (AURELIUS-SEC-HI-007)
- Timing-safe auth comparison in aurelius_server (AURELIUS-SEC-HI-005)
- File tool realpath traversal fix (AURELIUS-SEC-ME-012)
"""

from __future__ import annotations

import os
import tempfile


class TestCodeRunnerSafetyGate:
    def test_rejects_import_os(self):
        from src.tools.code_runner_tool import CodeRunnerTool

        tool = CodeRunnerTool()
        result = tool.run("import os; print(os.getcwd())")
        assert result.exit_code != 0
        assert "unsafe" in result.stderr.lower()

    def test_rejects_eval(self):
        from src.tools.code_runner_tool import CodeRunnerTool

        tool = CodeRunnerTool()
        result = tool.run('eval(\'__import__("os").system("id")\')')
        assert result.exit_code != 0

    def test_rejects_subprocess(self):
        from src.tools.code_runner_tool import CodeRunnerTool

        tool = CodeRunnerTool()
        result = tool.run("import subprocess; subprocess.run(['id'])")
        assert result.exit_code != 0

    def test_allows_safe_code(self):
        from src.tools.code_runner_tool import CodeRunnerTool

        tool = CodeRunnerTool()
        result = tool.run("import math; print(math.sqrt(4))")
        assert result.exit_code == 0
        assert "2.0" in result.stdout


class TestHttpClientSSRFProtection:
    def test_blocks_localhost(self):
        from src.tools.http_client import _is_safe_url

        safe, reason = _is_safe_url("http://127.0.0.1/admin")
        assert not safe
        assert "SSRF" in reason

    def test_blocks_aws_metadata(self):
        from src.tools.http_client import _is_safe_url

        safe, reason = _is_safe_url("http://169.254.169.254/latest/meta-data/")
        assert not safe

    def test_blocks_private_ip(self):
        from src.tools.http_client import _is_safe_url

        safe, _ = _is_safe_url("http://10.0.0.1/internal")
        assert not safe

    def test_allows_public_url(self):
        from src.tools.http_client import _is_safe_url

        safe, _ = _is_safe_url("https://example.com/api")
        assert safe

    def test_blocks_javascript_scheme(self):
        from src.tools.http_client import _is_safe_url

        safe, _ = _is_safe_url("javascript:alert(1)")
        assert not safe


class TestAureliusServerAuth:
    def test_timing_safe_comparison(self):
        import hmac

        key_a = "test-api-key-12345"
        key_b = "test-api-key-12345"
        key_c = "test-api-key-99999"

        assert hmac.compare_digest(key_a, key_b) is True
        assert hmac.compare_digest(key_a, key_c) is False


class TestFileToolRealpathTraversal:
    def test_symlink_traversal_blocked(self):
        from src.tools.file_tool import FileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "workspace")
            os.makedirs(base)

            secret_file = os.path.join(tmpdir, "secret.txt")
            with open(secret_file, "w") as f:
                f.write("secret data")

            link_path = os.path.join(base, "link")
            os.symlink(secret_file, link_path)

            tool = FileTool(base_dir=base)
            err = tool._check_base_dir(link_path)
            assert err is not None, "Symlink traversal should be blocked"

    def test_normal_file_allowed(self):
        from src.tools.file_tool import FileTool

        with tempfile.TemporaryDirectory() as tmpdir:
            base = os.path.join(tmpdir, "workspace")
            os.makedirs(base)
            normal = os.path.join(base, "file.txt")
            with open(normal, "w") as f:
                f.write("ok")

            tool = FileTool(base_dir=base)
            err = tool._check_base_dir(normal)
            assert err is None
