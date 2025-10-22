
from email import message_from_string
import io
from pathlib import Path

from markitdown import MarkItDown, StreamInfo

def extract_html_from_mhtml_file(mhtml_file: str) -> str | None:
    def _extract_html_from_mhtml(mhtml_content: str) -> str | None:
        """Extract the largest HTML content from MHTML string."""
        # hacky way to parse MHTML content and get the largest HTML part
        # This is a simplified example; a more robust implementation would be needed for production use.
        msg = message_from_string(mhtml_content)
        largest_html = None
        largest_size = 0
        for part in msg.walk():
            if part.get_content_type() == 'text/html':
                payload = part.get_payload(decode=True)
                if payload:
                    html = payload.decode('utf-8')
                    if len(html) > largest_size:
                        largest_size = len(html)
                        largest_html = html
        
        return largest_html
    
    with open(mhtml_file, 'r', encoding='utf-8') as f:
        mhtml_content = f.read()
        
    return _extract_html_from_mhtml(mhtml_content)  


def file_to_markdown(filename: str) -> str:
    """Convert a file to markdown using """
    md = MarkItDown(enable_plugins=True)
    if filename.endswith((".mhtml", ".mht")):
        content = extract_html_from_mhtml_file(filename)
        if content is None:
            raise ValueError("No HTML content found in MHTML file.")
        return md.convert(io.BytesIO(content.encode('utf-8')), stream_info=StreamInfo(extension=".html")).markdown

    return md.convert(Path(filename)).markdown

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils.py <file>")
        sys.exit(1)

    filename = sys.argv[1]
    if filename.endswith((".mhtml", ".mht")):
        content = extract_html_from_mhtml_file(filename)
        print(content)
        sys.exit(0)

    markdown_content = file_to_markdown(filename)
    print(markdown_content)