import pymupdf
import base64


class PDFLoader:

    def __init__(self, file_path):
        """Extracting content from PDF files including text, tables, and images."""

        self.file_path = file_path
        self.doc = pymupdf.open(file_path)

    def _extract_texts_tables_from_page(self, pno: int) -> str | None:
        """
        Extract text and tables from a specific page, avoiding text duplication in table areas.
        Args:
            pno (int): Page number (0-indexed)
        """
        page = self.doc[pno]
        tables = page.find_tables()

        if not tables:
            return page.get_text(), []

        page_tables = [
            {"table": n + 1, "data": tb.extract()} for n, tb in enumerate(tables)
        ]

        # avoid tables extraction
        exclude = [pymupdf.Rect(tab.bbox) for tab in tables]

        # Build clip areas that avoid all table bboxes
        clips = []
        y_start = 0
        for rect in sorted(exclude, key=lambda r: r.y0):
            clips.append(pymupdf.Rect(0, y_start, page.rect.width, rect.y0))
            y_start = rect.y1
        clips.append(pymupdf.Rect(0, y_start, page.rect.width, page.rect.height))

        # Extract text from non-table zones
        plain = ""
        for clip in clips:
            plain += page.get_text(clip=clip)
        return plain, page_tables

    def _extract_images_from_page(self, pno: int) -> list[dict[str, any]] | None:
        """Extract all images from a specific page and convert them to base64 format."""
        page = self.doc[pno]
        images_refs = page.get_images(full=True)

        imgs = []

        for n, img in enumerate(images_refs):
            img_data = self.doc.extract_image(img[0])
            # Convert bytes to base64 string to make it JSON serializable
            image_b64 = base64.b64encode(img_data["image"]).decode()
            imgs.append(
                {
                    "image_id": n + 1,
                    "base64": image_b64,  # Now it's a string, not bytes
                    "ext": img_data["ext"],
                }
            )

        return imgs

    def analyse(self):
        """
        Analyze the entire PDF document and extract all content.
        Returns List of dictionaries, one per page, containing: page number, plain text, tables, images
        """
        pages = []
        for p in self.doc:
            text_and_tables = self._extract_texts_tables_from_page(p.number)
            page_data = {
                "page": p.number + 1,
                "plain_text": text_and_tables[0],
                "tables": text_and_tables[1],
                "images": self._extract_images_from_page(p.number),
            }
            pages.append(page_data)
        return pages
