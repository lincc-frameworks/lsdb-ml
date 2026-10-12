"""Display basic version information to standard out."""

from lsdb_ml._version import __version__


def main():
    """The main function for the example file."""
    print(f"lsdb_ml: {__version__}")


if __name__ == "__main__":
    main()
