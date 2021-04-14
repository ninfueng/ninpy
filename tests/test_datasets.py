from ninpy.datasets.camvid import camvid_labels


def test_camvid_labels():
    list_train, list_val, list_test = camvid_labels()
    assert len(list_train) == 367
    assert len(list_val) == 101
    assert len(list_test) == 233
