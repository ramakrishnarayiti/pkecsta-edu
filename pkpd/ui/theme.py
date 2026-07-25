"""App-wide QSS stylesheet — ported from the "PKPD Studio" design mockup
(Classical design system: warm neutral surfaces, forest-green accent,
serif headings). Source tokens (light theme, default green accent variant):

bg #f3f2f2  surface #eae9e9  surface2(hover) #f8f4f4  text #201f1d
divider rgba(32,31,29,0.16)  accent #2F8F5B  accent-text #20613e (darker
green, used for active-state text — accent itself is reserved for fills/
borders/icons per the source design)
status: success #16A34A  warning #B45309  error #DC2626

Fonts: source design uses Cormorant Garamond (headings) / Lora (body) from
Google Fonts — not bundled (no font files shipped, only a CDN @import), so
this falls back to system serif faces. Component chrome (inputs, nav items,
table headers) keeps the source's actual in-app sizes: 13px inputs/nav,
12.5px buttons, 10.5px uppercase table headers — not the marketing-page
scale (42px h1 etc.), which belongs to a landing page this app doesn't have.
Data tables stay monospace (Consolas) for numeric alignment — the source
mockup never renders a real dense numeric grid, so this is a deliberate
adaptation, not a deviation.
"""

STYLESHEET = """
QMainWindow, QWidget {
    background-color: #f3f2f2;
    color: #201f1d;
    font-family: "Lora", "Constantia", "Georgia", serif;
    font-size: 13px;
}

QMainWindow::separator {
    background: #d1d0d0;
}

/* Sidebar nav list (QListWidget#sidebarNav) — mirrors the source design's
   left nav list (.pk-navitem / .pk-navitem-active). */
QListWidget#sidebarNav {
    background: #eae9e9;
    border: none;
    border-right: 1px solid #d1d0d0;
    padding: 8px 8px;
    outline: none;
}

QListWidget#sidebarNav::item {
    padding: 7px 10px;
    margin: 1px 0;
    border-radius: 4px;
    color: #201f1d;
}

QListWidget#sidebarNav::item:hover:!selected {
    background: #f8f4f4;
}

QListWidget#sidebarNav::item:selected {
    background: #d0dcd5;
    color: #20613e;
    font-weight: 600;
}

QPushButton {
    background: transparent;
    color: #201f1d;
    border: 1px solid #d1d0d0;
    padding: 6px 14px;
    border-radius: 4px;
    font-family: "Cormorant Garamond", "Constantia", "Georgia", serif;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background: #f8f4f4;
}

QPushButton:pressed {
    background: #d0dcd5;
    border-color: #2F8F5B;
}

QPushButton:disabled {
    color: #a8a5a3;
    border-color: #e2e0e0;
}

QComboBox, QLineEdit {
    background: white;
    color: #201f1d;
    border: 1px solid #d1d0d0;
    border-radius: 3px;
    padding: 3px 6px;
    min-height: 22px;
    selection-background-color: #2F8F5B;
}

QComboBox:focus, QLineEdit:focus {
    border: 1px solid #2F8F5B;
}

QComboBox QAbstractItemView {
    background: white;
    color: #201f1d;
    border: 1px solid #d1d0d0;
    selection-background-color: #2F8F5B;
    selection-color: white;
    outline: none;
}

QCheckBox {
    spacing: 6px;
}

QTableWidget {
    background: white;
    color: #201f1d;
    gridline-color: #ececea;
    border: 1px solid #d1d0d0;
    selection-background-color: #d0dcd5;
    selection-color: #20613e;
    font-family: "Consolas", "Cascadia Mono", monospace;
    font-size: 12px;
}

QHeaderView::section {
    background-color: #eae9e9;
    color: #5b5856;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #d1d0d0;
    font-family: "Lora", "Constantia", serif;
    font-weight: 600;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QTableWidget::item {
    padding: 2px 4px;
}

QLabel {
    color: #201f1d;
}

QSplitter::handle {
    background: #d1d0d0;
}

QTextBrowser {
    background: #f8f4f4;
    color: #201f1d;
    border: 1px solid #d1d0d0;
    padding: 14px;
}

QTextBrowser h2, QTextBrowser h3 {
    color: #20613e;
    font-family: "Cormorant Garamond", "Constantia", "Georgia", serif;
}

QTextBrowser table {
    border: 1px solid #d1d0d0;
}

QTextBrowser code {
    background: #eae9e9;
    color: #20613e;
    padding: 1px 4px;
    border-radius: 3px;
}

QScrollBar:vertical {
    background: #f3f2f2;
    width: 12px;
}

QScrollBar::handle:vertical {
    background: #d1d0d0;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #2F8F5B;
}

QScrollBar:horizontal {
    background: #f3f2f2;
    height: 12px;
}

QScrollBar::handle:horizontal {
    background: #d1d0d0;
    border-radius: 5px;
    min-width: 24px;
}

QScrollBar::handle:horizontal:hover {
    background: #2F8F5B;
}

QMessageBox {
    background: #f3f2f2;
}
"""
