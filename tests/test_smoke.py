def test_package_imports():
    import ml_training  # noqa: F401
    from ml_training import config, train  # noqa: F401