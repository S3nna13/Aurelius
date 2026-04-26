from src.security.report_generator import FindingEntry, ReportConfig, ReportGenerator


class TestReportGenerator:
    def test_empty_report(self):
        rg = ReportGenerator()
        stats = rg.summary_stats()
        assert stats["CRITICAL"] == 0
        assert stats["HIGH"] == 0

    def test_add_finding(self):
        rg = ReportGenerator()
        rg.add_finding(FindingEntry(title="SQL Injection", severity="HIGH", cwe_id="CWE-89"))
        assert rg.summary_stats()["HIGH"] == 1

    def test_add_findings_bulk(self):
        rg = ReportGenerator()
        rg.add_findings(
            [
                FindingEntry(title="XSS", severity="MEDIUM"),
                FindingEntry(title="CSRF", severity="LOW"),
            ]
        )
        assert rg.summary_stats()["MEDIUM"] == 1
        assert rg.summary_stats()["LOW"] == 1

    def test_to_markdown(self):
        rg = ReportGenerator(ReportConfig(title="Test Report", author="Tester"))
        rg.add_finding(FindingEntry(title="Test Finding", severity="HIGH", description="A test"))
        md = rg.to_markdown()
        assert "# Test Report" in md
        assert "Test Finding" in md
        assert "HIGH" in md
        assert "A test" in md

    def test_to_json(self):
        rg = ReportGenerator()
        rg.add_finding(FindingEntry(title="JSON Finding", severity="MEDIUM"))
        data = rg.to_json()
        assert "JSON Finding" in data
        assert "MEDIUM" in data

    def test_to_html(self):
        rg = ReportGenerator(ReportConfig(title="HTML Report"))
        rg.add_finding(FindingEntry(title="HTML Finding", severity="LOW"))
        html = rg.to_html()
        assert "HTML Report" in html
        assert "HTML Finding" in html
        assert "<!DOCTYPE html>" in html

    def test_clear(self):
        rg = ReportGenerator()
        rg.add_finding(FindingEntry(title="Temp", severity="INFO"))
        assert rg.summary_stats()["INFO"] == 1
        rg.clear()
        assert rg.summary_stats()["INFO"] == 0

    def test_custom_config(self):
        cfg = ReportConfig(title="Custom", author="Me", company="ACME")
        rg = ReportGenerator(cfg)
        assert rg.config.title == "Custom"
        assert rg.config.company == "ACME"

    def test_finding_entry_defaults(self):
        f = FindingEntry(title="Test", severity="LOW")
        assert f.cwe_id == ""
        assert f.cvss_score == 0.0
