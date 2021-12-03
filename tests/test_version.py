from ninpy import extract_version


def test_extract_version():
    major, minor, patch = extract_version("1.9.1+cu")
    assert major == 1
    assert minor == 9
    assert patch == 1
