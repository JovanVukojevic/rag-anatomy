from importlib.resources import files


def test_package_ships_py_typed_marker() -> None:
    assert files("rag_anatomy").joinpath("py.typed").is_file()
