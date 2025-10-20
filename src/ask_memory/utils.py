
from email import message_from_string

from markitdown import MarkItDown

def extract_html_from_mhtml(mhtml_file: str) -> str | None:
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


if __name__ == "__main__":
    # get file path from command line argument
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils.py <mhtml_file>")
        sys.exit(1)

    # if extension is .mhtml or .mht
    if sys.argv[1].endswith((".mhtml", ".mht")):
        html_content = extract_html_from_mhtml(sys.argv[1])
        if html_content:
            with open("output.html", "w", encoding="utf-8") as f:
                f.write(html_content)
            print("Extracted HTML content saved to output.html")
        else:
            print("No HTML content found in the MHTML file.")
        exit(0)
    
    md = MarkItDown(enable_plugins=True).convert(sys.argv[1])
    print(md.markdown)
    # with open("output.md", "w", encoding="utf-8") as f:
    #     f.write(md.markdown)
    # print("Converted Markdown content saved to output.md")
    exit(0)