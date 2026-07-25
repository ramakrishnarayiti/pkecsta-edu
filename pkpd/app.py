"""PKecsta — entry point.

Developed by Rayiti Ramakrishna, PharmD, PhD candidate at CSIR-CDRI.
Built with the assistance of Claude (Anthropic).
"""
import sys

from PySide6.QtWidgets import QApplication

from pkpd.ui.main_window import MainWindow
from pkpd.ui.theme import STYLESHEET


def main() -> None:
    app = QApplication(sys.argv)
    # Fusion style makes the QSS fully authoritative — the native Windows
    # style bleeds its own accent-color selection decoration through QSS
    # item styling otherwise (a stray blue sliver on selected list/table rows).
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
