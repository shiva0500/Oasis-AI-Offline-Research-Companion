from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QTextEdit, QComboBox, QGroupBox
from PyQt6.QtGui import QIcon, QTextOption
import sys
from ai_processor import extract_text_from_pdf, chunk_text, vectorize_text, build_faiss_index, save_index_and_chunks, load_index_and_chunks, ask_ai_model, AVAILABLE_MODELS

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oasis AI: Offline Research Companion")
        self.setGeometry(100, 100, 800, 600)

        # Set application icon (Shown in title bar & taskbar)
        self.setWindowIcon(QIcon("assets/logo.png"))

        layout = QVBoxLayout()

        # File Upload Button
        self.upload_button = QPushButton("📂 Upload PDF")
        self.upload_button.clicked.connect(self.upload_pdf)
        layout.addWidget(self.upload_button)

        # Model Selection
        self.model_selector = QComboBox()
        self.model_selector.addItems(AVAILABLE_MODELS)
        layout.addWidget(self.model_selector)

        # Question Input
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText("💡 Type your question here...")
        self.question_input.setFixedHeight(80)  # Fixed height to avoid taking too much space
        layout.addWidget(self.question_input)

        # Ask AI Button
        self.ask_button = QPushButton("🤖 Ask AI")
        self.ask_button.clicked.connect(self.ask_ai)
        layout.addWidget(self.ask_button)

        # Response Display (Multiline)
        self.response_label = QTextEdit()
        self.response_label.setPlaceholderText("🧠 AI Response will appear here.")
        self.response_label.setReadOnly(True)  # Make response label read-only
        self.response_label.setWordWrapMode(QTextOption.WrapMode.WordWrap)  # Correct way to enable word wrapping
        layout.addWidget(self.response_label)

        # Container to hold the layout
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def upload_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PDF File", "", "PDF Files (*.pdf)")
        if file_path:
            self.process_pdf(file_path)

    def process_pdf(self, file_path):
        self.response_label.setText("📄 Processing PDF...")
        pdf_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(pdf_text)
        embeddings = vectorize_text(chunks)
        index = build_faiss_index(embeddings)
        save_index_and_chunks(index, chunks)
        self.response_label.setText("✅ PDF processed and indexed!")

    def ask_ai(self):
        question = self.question_input.toPlainText()
        model_name = self.model_selector.currentText()
        index, chunks = load_index_and_chunks()
        if index and chunks:
            answer = ask_ai_model(question, chunks, index, model_name)
            self.response_label.setText(f"🧠 AI Response:\n{answer}")
        else:
            self.response_label.setText("⚠ No indexed PDF found.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
