"""Entry point for `python -m buzz.setup`: launches the terminal setup program."""

from buzz.setup.app import SetupApp

if __name__ == '__main__':  # pragma: no cover
    SetupApp().run()
