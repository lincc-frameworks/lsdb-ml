import lsdb_ml


def test_version():
    """Check to see that we can get the package version"""
    assert lsdb_ml.__version__ is not None
