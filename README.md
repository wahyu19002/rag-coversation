Aplikasi berbasis Streamlit yang memungkinkan pengguna untuk mengunggah satu atau lebih file PDF 
dan berinteraksi dengan isinya melalui percakapan berbasis AI. Sistem ini menggunakan teknik RAG (Retrieval-Augmented Generation)
untuk memberikan jawaban yang akurat berdasarkan konteks dokumen yang diunggah.

**Fitur Utama**
- Multi-PDF Upload: Mendukung pengunggahan beberapa file PDF sekaligus.
- Context-Aware Chat: AI memahami riwayat percakapan sehingga pengguna dapat mengajukan pertanyaan tindak lanjut tanpa mengulang konteks.
- Session Management: Menggunakan Session ID untuk memisahkan riwayat percakapan antar sesi.
- State-of-the-Art LLM: Didukung oleh model Llama 3.3-70b melalui Groq API untuk inferensi.
- Local Embeddings: Menggunakan model all-MiniLM-L6-v2 dari HuggingFace untuk mengubah teks menjadi vektor secara efisien.

**Teknologi yang digunakan**
- Framework : Streamlit
- Orchestration: LangChain
- LLM: Groq Cloud (Llama 3.3)
- Vector Store: ChromaDB
- Embeddings: HuggingFace
- PDF Processing: PyPDFLoader

**COBA DEMO APLIKASI DISINI**
https://rag-conversation-u8k2jksjyk8nbycqiccxdb.streamlit.app/
