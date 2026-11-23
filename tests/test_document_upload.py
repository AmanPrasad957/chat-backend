"""Automated test suite verifying document and file upload (PDF, TXT, CSV), chunking, and ChromaDB vector search."""

import io
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_in_memory_pdf(text: str) -> bytes:
    """Helper to generate a valid minimal in-memory PDF with text readable by pypdf."""
    content = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET"
    pdf = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {len(content)} >>
stream
{content}
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000240 00000 n 
0000000350 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
450
%%EOF"""
    return pdf.encode("latin-1")


def test_document_upload_suite():
    """Verify end-to-end document upload, multi-format parsing, chunking, and ChromaDB retrieval."""
    created_doc_ids = []

    try:
        # 1. Test Text File Upload
        txt_content = (
            "Acme Solar Battery 5000 Warranty Policy.\n\n"
            "All solar storage batteries purchased after January 2026 include a comprehensive 10-year replacement warranty.\n\n"
            "Degradation under 80% original capacity within 7 years will trigger immediate free replacement by our technicians."
        ).encode("utf-8")

        txt_resp = client.post(
            "/knowledge/upload",
            files={"file": ("battery_warranty.txt", io.BytesIO(txt_content), "text/plain")},
            data={"category": "solar_warranty", "chunk_size": 250, "chunk_overlap": 40},
        )
        assert txt_resp.status_code == 201, f"Text upload failed: {txt_resp.text}"
        txt_data = txt_resp.json()
        assert txt_data["status"] == "success"
        assert txt_data["filename"] == "battery_warranty.txt"
        assert txt_data["total_chunks"] >= 1
        created_doc_ids.extend(txt_data["doc_ids"])
        print(f"✓ Uploaded TXT file into {txt_data['total_chunks']} chunk(s).")

        # 2. Test CSV File Upload
        csv_content = (
            "ProductID,ProductName,Price,StockLevel,WarrantyYears\n"
            "SOL-101,Solar Inverter 5kW,1200,45,5\n"
            "BAT-202,Lithium Power Pack 10kWh,3500,18,10\n"
            "PNL-303,Monocrystalline Panel 400W,220,120,25\n"
        ).encode("utf-8")

        csv_resp = client.post(
            "/knowledge/upload",
            files={"file": ("inventory_catalog.csv", io.BytesIO(csv_content), "text/csv")},
            data={"category": "products"},
        )
        assert csv_resp.status_code == 201, f"CSV upload failed: {csv_resp.text}"
        csv_data = csv_resp.json()
        assert csv_data["status"] == "success"
        assert csv_data["file_type"] == "csv"
        created_doc_ids.extend(csv_data["doc_ids"])
        print(f"✓ Uploaded CSV file into {csv_data['total_chunks']} chunk(s).")

        # 3. Test PDF File Upload
        pdf_bytes = create_in_memory_pdf("Quantum Ultra Drone has an operational flight time of 48 minutes per battery charge.")
        pdf_resp = client.post(
            "/knowledge/upload",
            files={"file": ("drone_specs.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"category": "drones"},
        )
        assert pdf_resp.status_code == 201, f"PDF upload failed: {pdf_resp.text}"
        pdf_data = pdf_resp.json()
        assert pdf_data["status"] == "success"
        assert pdf_data["file_type"] == "pdf"
        created_doc_ids.extend(pdf_data["doc_ids"])
        print(f"✓ Uploaded PDF file into {pdf_data['total_chunks']} chunk(s).")

        # 4. Semantic Search Verification in ChromaDB
        search_resp = client.post(
            "/knowledge/query",
            json={"query": "How many years is the solar battery replacement warranty?", "n_results": 3},
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert search_data["results_count"] > 0
        texts = [r["text"] for r in search_data["results"]]
        assert any("Solar Battery" in t or "warranty" in t.lower() for t in texts), "ChromaDB should retrieve uploaded solar warranty"
        print("✓ Verified ChromaDB semantic vector search retrieves uploaded content.")

        # 5. Unsupported file format rejection
        bad_resp = client.post(
            "/knowledge/upload",
            files={"file": ("script.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
        )
        assert bad_resp.status_code == 400
        assert "Unsupported file format" in bad_resp.json()["detail"]
        print("✓ Unsupported file format correctly rejected with 400 Bad Request.")

        # 6. Empty file rejection
        empty_resp = client.post(
            "/knowledge/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )
        assert empty_resp.status_code == 400
        assert "empty" in empty_resp.json()["detail"].lower()
        print("✓ Empty file correctly rejected with 400 Bad Request.")

    finally:
        # Clean up created docs from ChromaDB
        for doc_id in created_doc_ids:
            try:
                client.delete(f"/knowledge/{doc_id}")
            except Exception:
                pass
        print(f"✓ Cleaned up {len(created_doc_ids)} test document chunk(s) from ChromaDB.")


if __name__ == "__main__":
    test_document_upload_suite()
    print("ALL DOCUMENT UPLOAD & CHUNKING TESTS PASSED SUCCESSFULLY!")
