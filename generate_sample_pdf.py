from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_pdf(filename, content):
    c = canvas.Canvas(filename, pagesize=letter)
    textobject = c.beginText()
    textobject.setTextOrigin(50, 750)
    textobject.setFont("Helvetica", 12)
    for line in content.split("\n"):
        textobject.textLine(line)
    c.drawText(textobject)
    c.save()

if __name__ == "__main__":
    sample_content = """
    ## The Multi-RAG System: A Visionary Approach to Information Retrieval

    The Multi-RAG (Retrieval-Augmented Generation) system represents a significant leap forward in how we interact with vast amounts of information. Unlike traditional search engines that merely return links, or simple RAG systems that rely on a single source, Multi-RAG intelligently synthesizes information from diverse, multimodal sources to provide comprehensive and accurate answers.

    ### Key Features:

    *   **Multimodal Data Ingestion:** Processes text, images, and tabular data from various document formats (PDFs, Word documents, web pages).
    *   **Advanced Retrieval:** Employs sophisticated indexing and vector search techniques to identify the most relevant pieces of information across different modalities.
    *   **Contextual Generation:** Leverages large language models (LLMs) to generate human-like responses, augmented by the retrieved context, ensuring factual accuracy and reducing hallucinations.
    *   **Real-time Processing:** Designed for low-latency query processing and streaming responses, making it suitable for interactive applications.
    *   **Scalable Architecture:** Built with a modular and scalable design, utilizing PostgreSQL for metadata, Milvus for vector storage, and Redis for caching and task queuing.

    ### Use Cases:

    1.  **Enterprise Knowledge Management:** Quickly find answers across internal documents, reports, and databases.
    2.  **Customer Support Automation:** Provide accurate and consistent answers to customer queries by drawing from product manuals and FAQs.
    3.  **Research and Development:** Accelerate research by synthesizing information from academic papers, patents, and technical specifications.
    4.  **Legal Discovery:** Efficiently analyze large volumes of legal documents to identify relevant precedents and facts.

    This system is designed to be deployment-ready, with a robust backend, a user-friendly web interface, and comprehensive testing. It aims to transform information access and utilization across various industries.
    """
    generate_pdf("/home/ubuntu/multi-rag-system/data/uploads/sample_document.pdf", sample_content)
