"""Desktop application entry point."""

from .gui import ArnoldViewer


def main() -> None:
    ArnoldViewer().run()


if __name__ == "__main__":
    main()
