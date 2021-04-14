from ninpy.datasets.camvid import camvid_dirs


def test_camvid_dirs():
    list_train, list_val, list_test = camvid_dirs()
    assert len(list_train) == 367
    assert len(list_val) == 101
    assert len(list_test) == 233
