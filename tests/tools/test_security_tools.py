from src.tools.security_tools import SecurityToolManager, ToolResult


class TestSecurityToolManager:
    def test_discover_tools(self):
        mgr = SecurityToolManager()
        assert isinstance(mgr.available_tools(), list)

    def test_is_available(self):
        mgr = SecurityToolManager()
        result = mgr.is_available("nonexistent_tool_xyz")
        assert not result

    def test_run_nmap_not_found(self):
        mgr = SecurityToolManager({})
        result = mgr.run_nmap("127.0.0.1")
        assert result.error is not None

    def test_run_nikto_not_found(self):
        mgr = SecurityToolManager({})
        result = mgr.run_nikto("http://localhost")
        assert result.error is not None

    def test_run_sqlmap_not_found(self):
        mgr = SecurityToolManager({})
        result = mgr.run_sqlmap("http://localhost/test")
        assert result.error is not None

    def test_tool_result_defaults(self):
        tr = ToolResult(tool="nmap")
        assert tr.return_code == -1
        assert tr.error is None
