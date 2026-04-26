"""Unit tests for IOCExtractor."""

from src.security.ioc_extractor import IOCExtractor, refang, validate_bitcoin_address


def test_ipv4_extracted():
    ex = IOCExtractor(include_private_ips=True)
    rep = ex.extract("attacker at 1.2.3.4 and 10.0.0.1")
    types = [i.type for i in rep.iocs]
    assert "ipv4" in types


def test_private_ip_excluded_by_default():
    ex = IOCExtractor()
    rep = ex.extract("private 192.168.1.1")
    vals = [i.value for i in rep.iocs if i.type == "ipv4"]
    assert "192.168.1.1" not in vals


def test_include_private_ips_true():
    ex = IOCExtractor(include_private_ips=True)
    rep = ex.extract("private 192.168.1.1")
    vals = [i.value for i in rep.iocs if i.type == "ipv4"]
    assert "192.168.1.1" in vals


def test_google_com_excluded_by_default():
    ex = IOCExtractor()
    rep = ex.extract("benign google.com site")
    vals = [i.value for i in rep.iocs if i.type == "domain"]
    assert "google.com" not in vals


def test_url_extracted():
    ex = IOCExtractor()
    rep = ex.extract("see https://evil-bad-domain.example/malware")
    assert any(i.type == "url" for i in rep.iocs)


def test_email_extracted():
    ex = IOCExtractor()
    rep = ex.extract("contact attacker@evil-domain-xyz.net for details")
    assert any(i.type == "email" for i in rep.iocs)


def test_md5_hash_extracted():
    h = "d41d8cd98f00b204e9800998ecf8427e"
    ex = IOCExtractor()
    rep = ex.extract(f"md5 hash: {h}")
    assert any(i.type == "md5" and i.value.lower() == h for i in rep.iocs)


def test_sha256_extracted():
    h = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ex = IOCExtractor()
    rep = ex.extract(f"sha256: {h}")
    assert any(i.type == "sha256" for i in rep.iocs)


def test_cve_extracted():
    ex = IOCExtractor()
    rep = ex.extract("Log4Shell CVE-2021-44228 exploit")
    assert any(i.type == "cve" and i.value == "CVE-2021-44228" for i in rep.iocs)


def test_windows_path_detected():
    ex = IOCExtractor()
    rep = ex.extract(r"C:\Windows\System32\evil.exe")
    assert any(i.type in ("windows_path", "file_path", "file_path_windows") for i in rep.iocs)


def test_registry_key_detected():
    ex = IOCExtractor()
    rep = ex.extract(r"persistence via HKLM\Software\Microsoft\Windows\CurrentVersion\Run")
    assert any(i.type == "registry_key" for i in rep.iocs)


def test_defanged_url_refanged():
    ex = IOCExtractor(refang=True)
    rep = ex.extract("hxxp://malicious-xyz[.]com/payload")
    urls = [i for i in rep.iocs if i.type == "url"]
    assert any("http" in u.value for u in urls)


def test_refang_function_idempotent():
    s1 = refang("hxxp://x[.]com")
    s2 = refang(s1)
    assert s1 == s2


def test_validate_bitcoin_invalid_rejected():
    # non-base58 gibberish
    assert not validate_bitcoin_address("not-a-real-address")
    # garbage checksum (random-looking but base58-valid chars)
    assert not validate_bitcoin_address("1AAAbbbCCCdddEEEfffGGGhhhIIIjjj")


def test_empty_text_returns_empty_report():
    ex = IOCExtractor()
    rep = ex.extract("")
    assert rep.total == 0
    assert rep.iocs == []


def test_determinism():
    t = "attacker@evil-xyz.net from 1.2.3.4 at https://bad-zzz.example"
    r1 = IOCExtractor(include_private_ips=True).extract(t)
    r2 = IOCExtractor(include_private_ips=True).extract(t)
    assert [(i.type, i.value) for i in r1.iocs] == [(i.type, i.value) for i in r2.iocs]


def test_ioc_report_has_by_type_dict():
    ex = IOCExtractor()
    rep = ex.extract("CVE-2021-44228 and CVE-2024-1234")
    assert isinstance(rep.by_type, dict)
    assert "cve" in rep.by_type
