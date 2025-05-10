from html.parser import HTMLParser
import logging
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class HTMLValidator:
    """
    Validates and repairs HTML content using BeautifulSoup
    """
    def __init__(self, allowed_tags=None, forbidden_tags=None):
        # Added 'th' to allowed tags for tables
        self.allowed_tags = allowed_tags or ['p', 'span', 'u', 'ol', 'ul', 'li', 'table', 'tr', 'td', 'th', 'tbody', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        # Keep 'em', 'br' forbidden as they are handled or disallowed
        self.forbidden_tags = forbidden_tags or ['script', 'iframe', 'style', 'link', 'meta', 'head', 'body', 'html', 'div', 'em', 'br', 'i', 'b']

    def _clean_llm_html_output(self, raw_html: str) -> str:
        """Strips common LLM artifacts like markdown code fences."""
        logger.debug(f"Raw HTML before cleaning:\n{raw_html[:500]}...") # Log start of raw HTML
        # Remove potential markdown fences (html, xml etc.)
        cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*", "", raw_html)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        # Remove potential leading/trailing explanations
        cleaned = cleaned.strip()
        logger.debug(f"HTML after stripping fences/whitespace:\n{cleaned[:500]}...") # Log after stripping
        return cleaned

    def validate_and_repair(self, html_content: str, clean_llm_output: bool = False) -> str:
        """
        Validates and repairs HTML content

        Args:
            html_content: HTML content to validate and repair
            clean_llm_output: If True, first clean common LLM artifacts

        Returns:
            Validated and repaired HTML content
        """
        if not isinstance(html_content, str):
             logger.warning(f"Invalid input type for HTML validation: {type(html_content)}. Returning empty content.")
             return '<p><span>Invalid input</span></p>'

        try:
            if clean_llm_output:
                processed_html = self._clean_llm_html_output(html_content)
            else:
                processed_html = html_content

            if not processed_html.strip():
                 logger.warning("Empty HTML content after initial processing.")
                 return {"status": "error", "html": "Empty content"}

            # Parse the HTML using BeautifulSoup with 'html.parser'
            soup = BeautifulSoup(processed_html, 'html.parser')

            # --- Tag Replacement and Removal ---
            # Replace <i> with styled <span>
            for i_tag in soup.find_all('i'):
                span = soup.new_tag('span')
                span['style'] = 'font-style:italic;'
                span.extend(i_tag.contents) # Move children
                i_tag.replace_with(span)

            # Replace <b> with styled <span>
            for b_tag in soup.find_all('b'):
                span = soup.new_tag('span')
                span['style'] = 'font-weight:bold;'
                span.extend(b_tag.contents) # Move children
                b_tag.replace_with(span)

            # Remove forbidden tags (keep their contents using unwrap)
            # Iterate through a list copy as modifying the tree affects find_all
            tags_to_remove = []
            for tag_name in self.forbidden_tags:
                tags_to_remove.extend(soup.find_all(tag_name))

            for tag in tags_to_remove:
                 # Check if tag still exists in the tree before unwrapping
                 if tag.parent:
                     tag.unwrap()


            # --- Structure Repair ---
            # Ensure tables have tbody
            for table in soup.find_all('table'):
                # Check if tbody already exists
                if not table.find('tbody', recursive=False):
                    # Create tbody
                    tbody = soup.new_tag('tbody')
                    # Find direct child tr elements OR tr elements inside thead/tfoot if they exist wrongly
                    trs_to_move = table.find_all('tr', recursive=False)
                    thead = table.find('thead', recursive=False)
                    if thead:
                        trs_to_move.extend(thead.find_all('tr', recursive=True))
                        thead.extract() # Remove thead if exists wrongly
                    tfoot = table.find('tfoot', recursive=False)
                    if tfoot:
                        trs_to_move.extend(tfoot.find_all('tr', recursive=True))
                        tfoot.extract() # Remove tfoot if exists wrongly

                    if trs_to_move:
                        for tr in trs_to_move:
                            tbody.append(tr.extract()) # Move tr into tbody
                        table.append(tbody) # Add tbody to table
                    elif not table.contents or all(c.isspace() for c in table.contents):
                         # If table is empty or only whitespace after removing bad thead/tfoot
                         # Add an empty tbody to make it valid structurally
                         table.append(tbody)


            # Ensure text is inside spans (within allowed block tags)
            allowed_parent_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']
            for parent_tag_name in allowed_parent_tags:
                for parent in soup.find_all(parent_tag_name):
                    # Iterate through children carefully
                    contents_to_wrap = []
                    for content in parent.contents:
                         # Check if it's a NavigableString and not just whitespace
                        if isinstance(content, str) and content.strip():
                            contents_to_wrap.append(content)
                         # Also wrap direct children that are NOT allowed tags or already spans
                        elif content.name not in self.allowed_tags and content.name != 'span':
                             contents_to_wrap.append(content) # Treat disallowed tags like text to be wrapped

                    # Wrap the collected stray contents in spans
                    for item in contents_to_wrap:
                         span = soup.new_tag('span')
                         # Use replace_with to put the span in the item's place
                         # Handle both strings and tags
                         if isinstance(item, str):
                              span.string = item
                              item.replace_with(span)
                         elif item.name: # It's a tag
                             span.extend(item.contents) # Move content
                             item.replace_with(span) # Replace tag

            # --- Final Output Generation ---
            # Get the clean HTML from the body content if html/body tags were added by parser
            # otherwise, process the top-level elements
            if soup.body:
                clean_html = ''.join(str(child) for child in soup.body.contents)
            elif soup.html: # Handle case where only <html> tag might be present
                clean_html = ''.join(str(child) for child in soup.html.contents)
            else:
                clean_html = ''.join(str(child) for child in soup.contents)


            # If the HTML is completely empty after cleaning, provide a basic structure
            if not clean_html.strip():
                 logger.warning("HTML content became empty after validation. Providing default.")
                 # Check original input again - maybe it was just whitespace
                 if not html_content.strip():
                     return {"status": "error", "html": "Empty content"}
                 else:
                     # Original had content, but validator removed it all (maybe only forbidden tags?)
                     return {"status": "error", "html": "Content removed during validation"}


            logger.debug(f"Validated HTML:\n{clean_html[:500]}...")
            return {"status": "success", "html": clean_html.strip()}

        except Exception as e:
            logger.error(f"Critical error during HTML validation/repair: {e}", exc_info=True)
            # Return a safe fallback if parsing fails catastrophically
            # Escape the error message to prevent potential HTML injection in the error itself
            import html as html_escaper
            escaped_error = html_escaper.escape(str(e))
            return {"status": "error", "html": f"Content processing error: {escaped_error}"}
