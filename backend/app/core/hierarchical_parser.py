# backend/app/core/hierarchical_parser.py
"""
Hierarchical Document Parser using LayoutPDFReader
Extracts structured content from PDFs, HTML, Markdown, DOCX, PPTX
"""

import json
import os
from typing import Dict, List, Any, Optional
from llmsherpa.readers import LayoutPDFReader
import tempfile
import uuid

class HierarchicalDocumentParser:
    """Parser for extracting hierarchical structure from documents"""
    
    def __init__(self, llmsherpa_api_url: str = "http://127.0.0.1:5010/api/parseDocument?renderFormat=all&useNewIndentParser=true"):
        self.api_url = llmsherpa_api_url
        self.pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    
    def parse_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Parse a document and extract hierarchical structure
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename with extension
            
        Returns:
            Dictionary with hierarchical structure and metadata
        """
        try:
            # Determine file type
            file_ext = filename.lower().split('.')[-1] if '.' in filename else 'txt'
            
            if file_ext == 'pdf':
                return self._parse_pdf(file_content, filename)
            elif file_ext in ['html', 'htm']:
                return self._parse_html(file_content, filename)
            elif file_ext == 'md':
                return self._parse_markdown(file_content, filename)
            elif file_ext in ['docx', 'doc']:
                return self._parse_docx(file_content, filename)
            elif file_ext in ['pptx', 'ppt']:
                return self._parse_pptx(file_content, filename)
            else:
                return self._parse_text(file_content, filename)
                
        except Exception as e:
            print(f"Error parsing document {filename}: {str(e)}")
            # Fallback to simple text parsing
            return self._parse_text(file_content, filename)
    
    def _parse_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF using LayoutPDFReader"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Parse with LayoutPDFReader
            doc = self.pdf_reader.read_pdf(temp_file_path)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Extract hierarchical structure
            hierarchy = self._extract_hierarchy(doc)
            
            return {
                "success": True,
                "document_type": "pdf",
                "filename": filename,
                "hierarchy": hierarchy,
                "sections": self._extract_sections(doc),
                "tables": self._extract_tables(doc),
                "full_text": doc.to_text(),
                "raw_chunks": self._extract_raw_chunks(doc),  # NEW: Raw chunks with actual content
                "metadata": {
                    "total_sections": len(hierarchy.get("sections", [])),
                    "has_tables": len(self._extract_tables(doc)) > 0,
                    "parser": "LayoutPDFReader"
                }
            }
            
        except Exception as e:
            print(f"PDF parsing failed, falling back to text: {str(e)}")
            return self._parse_text(file_content, filename)
    
    def _parse_html(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Parse HTML document"""
        try:
            # For now, use simple text parsing
            # TODO: Implement proper HTML parsing with BeautifulSoup
            return self._parse_text(file_content, filename)
        except Exception as e:
            return self._parse_text(file_content, filename)
    
    def _parse_markdown(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Parse Markdown document"""
        try:
            text = file_content.decode('utf-8', errors='ignore')
            
            # Simple markdown hierarchy extraction
            lines = text.split('\n')
            hierarchy = {"title": "Document", "sections": []}
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    # Extract heading level and text
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('#').strip()
                    
                    if level == 1:
                        current_section = {"title": title, "sections": []}
                        hierarchy["sections"].append(current_section)
                    elif current_section and level == 2:
                        current_section["sections"].append({"title": title, "sections": []})
            
            return {
                "success": True,
                "document_type": "markdown",
                "filename": filename,
                "hierarchy": hierarchy,
                "sections": self._extract_markdown_sections(text),
                "tables": [],
                "full_text": text,
                "metadata": {
                    "total_sections": len(hierarchy.get("sections", [])),
                    "has_tables": False,
                    "parser": "markdown"
                }
            }
            
        except Exception as e:
            return self._parse_text(file_content, filename)
    
    def _parse_docx(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Parse DOCX document"""
        try:
            # For now, use simple text parsing
            # TODO: Implement proper DOCX parsing with python-docx
            return self._parse_text(file_content, filename)
        except Exception as e:
            return self._parse_text(file_content, filename)
    
    def _parse_pptx(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Parse PPTX document"""
        try:
            # For now, use simple text parsing
            # TODO: Implement proper PPTX parsing with python-pptx
            return self._parse_text(file_content, filename)
        except Exception as e:
            return self._parse_text(file_content, filename)
    
    def _parse_text(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Fallback text parsing"""
        try:
            text = file_content.decode('utf-8', errors='ignore')
            
            # Simple text hierarchy based on line patterns
            lines = text.split('\n')
            hierarchy = {"title": "Document", "sections": []}
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line and len(line) < 100:  # Potential heading
                    # Check if line looks like a heading
                    if (line.isupper() or 
                        line.endswith(':') or 
                        (len(line.split()) <= 8 and not line.endswith('.'))):
                        
                        if not current_section:
                            current_section = {"title": line, "sections": []}
                            hierarchy["sections"].append(current_section)
                        else:
                            current_section["sections"].append({"title": line, "sections": []})
            
            return {
                "success": True,
                "document_type": "text",
                "filename": filename,
                "hierarchy": hierarchy,
                "sections": self._extract_text_sections(text),
                "tables": [],
                "full_text": text,
                "metadata": {
                    "total_sections": len(hierarchy.get("sections", [])),
                    "has_tables": False,
                    "parser": "text"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Text parsing failed: {str(e)}",
                "document_type": "unknown",
                "filename": filename
            }
    
    def _extract_hierarchy(self, doc) -> Dict[str, Any]:
        """Extract hierarchical structure from LayoutPDFReader document"""
        try:
            nodes = []
            
            for sec in doc.sections():
                title = sec.title
                level = sec.level
                page_idx = sec.page_idx
                
                nodes.append({
                    "title": title,
                    "level": level,
                    "page_idx": page_idx,
                    "content": sec.to_text()
                })
            
            # Build hierarchy tree
            hierarchy = {"title": "Document", "sections": []}
            self._build_hierarchy_tree(nodes, hierarchy)
            
            return hierarchy
            
        except Exception as e:
            print(f"Error extracting hierarchy: {str(e)}")
            return {"title": "Document", "sections": []}
    
    def _build_hierarchy_tree(self, nodes: List[Dict], hierarchy: Dict):
        """Build hierarchical tree structure from flat nodes"""
        try:
            stack = [hierarchy]
            
            for node in nodes:
                level = node["level"]
                title = node["title"]
                content = node["content"]
                
                # Adjust stack to current level
                while len(stack) > level + 1:
                    stack.pop()
                
                # Create new section
                new_section = {
                    "title": title,
                    "content": content,
                    "sections": []
                }
                
                # Add to current parent
                if "sections" not in stack[-1]:
                    stack[-1]["sections"] = []
                stack[-1]["sections"].append(new_section)
                
                # Add to stack for next iteration
                stack.append(new_section)
                
        except Exception as e:
            print(f"Error building hierarchy tree: {str(e)}")
    
    def _extract_sections(self, doc) -> List[Dict[str, Any]]:
        """Extract sections with their actual chunk content"""
        try:
            sections = []
            
            # Method 1: Extract from sections (gives structure)
            for sec in doc.sections():
                # Get actual content from chunks within this section
                section_content = sec.to_text()
                
                sections.append({
                    "title": sec.title,
                    "level": sec.level,
                    "content": section_content,
                    "page_idx": sec.page_idx
                })
            
            # Method 2: Also extract chunks directly for better content coverage
            # This ensures we get paragraph content even if section structure is weak
            chunk_sections = self._extract_chunks_as_sections(doc)
            
            # Merge both approaches - prefer structured sections, but add chunks if needed
            if len(sections) < 3 and chunk_sections:
                print(f"[HierarchicalParser] Found only {len(sections)} sections, adding {len(chunk_sections)} chunk-based sections")
                sections.extend(chunk_sections)
            
            return sections
        except Exception as e:
            print(f"Error extracting sections: {str(e)}")
            return []
    
    def _extract_raw_chunks(self, doc) -> List[Dict[str, Any]]:
        """Extract raw chunks with actual content from llmsherpa doc.
        Creates larger, more contextual chunks by combining related paragraphs.
        """
        try:
            raw_chunks = []
            current_heading = "Introduction"
            current_content = []
            current_page = 0
            chunk_idx = 0
            
            for chk in doc.chunks():
                # Get actual text content
                chunk_text = "\n".join(chk.sentences) if hasattr(chk, 'sentences') else chk.to_text()
                
                # Skip empty chunks
                if not chunk_text or len(chunk_text.strip()) < 20:
                    continue
                
                # Detect if this is a heading
                is_heading = (len(chunk_text) < 150 and 
                             (hasattr(chk, 'tag') and chk.tag in ['heading', 'title', 'h1', 'h2', 'h3', 'section_header']) or
                             (len(chunk_text) < 80 and chunk_text.isupper()))
                
                if is_heading:
                    # Save accumulated content before switching sections
                    if current_content:
                        combined_text = "\n\n".join(current_content)
                        if len(combined_text.strip()) > 100:  # Only save substantial content
                            raw_chunks.append({
                                "text": combined_text.strip(),
                                "section_title": current_heading,
                                "page_idx": current_page,
                                "level": 0,
                                "chunk_index": chunk_idx
                            })
                            chunk_idx += 1
                        current_content = []
                    
                    # Update current heading context
                    current_heading = chunk_text.strip()
                    current_page = chk.page_idx if hasattr(chk, 'page_idx') else current_page
                else:
                    # Accumulate content under current heading
                    current_content.append(chunk_text.strip())
                    
                    # Create chunk when we have enough content (300-600 words)
                    combined_length = sum(len(c.split()) for c in current_content)
                    if combined_length > 200:  # ~200 words
                        combined_text = "\n\n".join(current_content)
                        raw_chunks.append({
                            "text": combined_text.strip(),
                            "section_title": current_heading,
                            "page_idx": chk.page_idx if hasattr(chk, 'page_idx') else current_page,
                            "level": chk.level if hasattr(chk, 'level') else 0,
                            "chunk_index": chunk_idx
                        })
                        chunk_idx += 1
                        # Keep last 50 words for overlap
                        last_sentence = current_content[-1] if current_content else ""
                        current_content = [last_sentence] if len(last_sentence.split()) > 20 else []
            
            # Save any remaining content
            if current_content:
                combined_text = "\n\n".join(current_content)
                if len(combined_text.strip()) > 100:
                    raw_chunks.append({
                        "text": combined_text.strip(),
                        "section_title": current_heading,
                        "page_idx": current_page,
                        "level": 0,
                        "chunk_index": chunk_idx
                    })
            
            print(f"[HierarchicalParser] Extracted {len(raw_chunks)} contextual chunks (avg ~200 words each)")
            return raw_chunks
            
        except Exception as e:
            print(f"Error extracting raw chunks: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def _extract_chunks_as_sections(self, doc) -> List[Dict[str, Any]]:
        """Extract chunks as pseudo-sections for better content coverage"""
        try:
            chunk_sections = []
            current_section = None
            
            for chk in doc.chunks():
                chunk_text = "\n".join(chk.sentences) if hasattr(chk, 'sentences') else chk.to_text()
                
                # Skip empty chunks
                if not chunk_text.strip() or len(chunk_text.strip()) < 30:
                    continue
                
                # Group chunks by page and level
                chunk_key = f"Page {chk.page_idx} - Level {chk.level}"
                
                # If this is a heading chunk (short text, specific tags), create new section
                if len(chunk_text) < 150 and chk.tag in ['heading', 'title', 'h1', 'h2', 'h3']:
                    if current_section and current_section.get('content'):
                        chunk_sections.append(current_section)
                    
                    current_section = {
                        "title": chunk_text.strip(),
                        "level": chk.level,
                        "content": "",
                        "page_idx": chk.page_idx,
                        "source": "chunk_extraction"
                    }
                else:
                    # This is content, add to current section
                    if current_section:
                        current_section["content"] += chunk_text + "\n\n"
                    else:
                        # No section yet, create a default one
                        current_section = {
                            "title": chunk_key,
                            "level": chk.level,
                            "content": chunk_text + "\n\n",
                            "page_idx": chk.page_idx,
                            "source": "chunk_extraction"
                        }
            
            # Add last section
            if current_section and current_section.get('content'):
                chunk_sections.append(current_section)
            
            print(f"[HierarchicalParser] Extracted {len(chunk_sections)} chunk-based sections")
            return chunk_sections
            
        except Exception as e:
            print(f"Error extracting chunks as sections: {str(e)}")
            return []
    
    def _extract_tables(self, doc) -> List[Dict[str, Any]]:
        """Extract tables from document"""
        try:
            tables = []
            for table in doc.tables():
                tables.append({
                    "content": table.to_text(),
                    "page_idx": table.page_idx
                })
            return tables
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []
    
    def _extract_markdown_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections from markdown text"""
        try:
            sections = []
            lines = text.split('\n')
            current_section = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('#').strip()
                    
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        "title": title,
                        "level": level,
                        "content": "",
                        "page_idx": 0
                    }
                elif current_section and line:
                    current_section["content"] += line + "\n"
            
            if current_section:
                sections.append(current_section)
            
            return sections
        except Exception as e:
            print(f"Error extracting markdown sections: {str(e)}")
            return []
    
    def _extract_text_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections from plain text"""
        try:
            sections = []
            lines = text.split('\n')
            current_section = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line and len(line) < 100 and not line.endswith('.'):
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        "title": line,
                        "level": 1,
                        "content": "",
                        "page_idx": 0
                    }
                elif current_section and line:
                    current_section["content"] += line + "\n"
            
            if current_section:
                sections.append(current_section)
            
            return sections
        except Exception as e:
            print(f"Error extracting text sections: {str(e)}")
            return []
    
    def get_document_outline(self, hierarchy: Dict[str, Any], max_depth: int = 4) -> str:
        """
        Extract deep document structure for LLM context.
        Returns a formatted hierarchical structure showing chapters, sections, subsections
        with context about content organization (excluding paragraph chunks).
        """
        try:
            outline_lines = []
            
            def extract_deep_outline(node, level=0, parent_path="", section_number=""):
                if level > max_depth:
                    return
                
                if "title" in node and node["title"] and node["title"] != "Document":
                    title = node["title"].strip()
                    
                    # Skip noise (page numbers, figure captions, etc.)
                    if (len(title) > 3 and len(title) < 200 and 
                        not title.lower().startswith(('page', 'figure', 'table', 'fig.', 'list of', 'contents'))):
                        
                        # Create hierarchical numbering
                        if level == 0:
                            current_number = f"{len(outline_lines) + 1}"
                        elif level == 1:
                            current_number = f"{section_number}.{len([s for s in outline_lines if s.count('.') == 0]) + 1}"
                        else:
                            current_number = f"{section_number}.{len([s for s in outline_lines if s.startswith(section_number)]) + 1}"
                        
                        # Build context path
                        current_path = f"{parent_path} > {title}" if parent_path else title
                        
                        # Get content preview (first 100 chars, no paragraphs)
                        content_preview = ""
                        if "content" in node and node["content"]:
                            content = node["content"].strip()
                            # Remove paragraph breaks and get first meaningful content
                            content_lines = [line.strip() for line in content.split('\n') if line.strip()]
                            if content_lines:
                                content_preview = content_lines[0][:100]
                                if len(content_lines[0]) > 100:
                                    content_preview += "..."
                        
                        # Format the outline entry
                        indent = "  " * level
                        outline_entry = f"{indent}{current_number}. {title}"
                        
                        # Add content context if available
                        if content_preview:
                            outline_entry += f"\n{indent}    └─ Content: {content_preview}"
                        
                        # Add structural context
                        subsections = node.get("sections", [])
                        if subsections:
                            outline_entry += f"\n{indent}    └─ Subsections: {len(subsections)}"
                        
                        outline_lines.append(outline_entry)
                        
                        # Update section number for next level
                        if level == 0:
                            next_section_number = current_number
                        else:
                            next_section_number = current_number
                        
                        # Recursively process subsections
                        for subsection in subsections:
                            extract_deep_outline(
                                subsection, 
                                level + 1, 
                                current_path, 
                                next_section_number
                            )
                    else:
                        # Still process subsections even if current title is noise
                        for subsection in node.get("sections", []):
                            extract_deep_outline(subsection, level, parent_path, section_number)
                else:
                    # Process subsections even if no title
                    for subsection in node.get("sections", []):
                        extract_deep_outline(subsection, level, parent_path, section_number)
            
            extract_deep_outline(hierarchy)
            
            if outline_lines:
                # Add document structure summary
                total_sections = len([line for line in outline_lines if not line.strip().startswith('  ')])
                total_subsections = len([line for line in outline_lines if line.strip().startswith('  ')])
                
                summary = f"""DOCUMENT STRUCTURE OVERVIEW:
Total Major Sections: {total_sections}
Total Subsections: {total_subsections}
Max Depth: {max_depth}

DETAILED STRUCTURE:
"""
                return summary + "\n".join(outline_lines)
            else:
                return "No clear document structure found"
            
        except Exception as e:
            print(f"Error extracting document outline: {str(e)}")
            return "Error extracting outline"
    
    def get_learning_roadmap(self, hierarchy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured learning roadmap from document hierarchy.
        Returns a roadmap that can be used for progress tracking and planning.
        """
        try:
            roadmap = {
                "chapters": [],
                "total_chapters": 0,
                "total_sections": 0,
                "estimated_learning_hours": 0
            }
            
            def build_roadmap(node, level=0, chapter_number=0, section_number=0):
                if "title" in node and node["title"] and node["title"] != "Document":
                    title = node["title"].strip()
                    
                    # Skip noise
                    if (len(title) > 3 and len(title) < 200 and 
                        not title.lower().startswith(('page', 'figure', 'table', 'fig.', 'list of', 'contents'))):
                        
                        if level == 0:  # Chapter level
                            chapter_number += 1
                            chapter = {
                                "chapter_id": f"ch_{chapter_number}",
                                "chapter_number": chapter_number,
                                "title": title,
                                "sections": [],
                                "estimated_hours": 0,
                                "status": "not_started"
                            }
                            roadmap["chapters"].append(chapter)
                            roadmap["total_chapters"] += 1
                            
                            # Process sections
                            for section in node.get("sections", []):
                                section_data = build_roadmap(section, level + 1, chapter_number, 0)
                                if section_data:
                                    chapter["sections"].append(section_data)
                                    roadmap["total_sections"] += 1
                            
                            # Estimate hours for chapter (2-4 hours per major section)
                            chapter["estimated_hours"] = max(2, len(chapter["sections"]) * 1.5)
                            roadmap["estimated_learning_hours"] += chapter["estimated_hours"]
                            
                        elif level == 1:  # Section level
                            section_number += 1
                            section = {
                                "section_id": f"ch_{chapter_number}_sec_{section_number}",
                                "section_number": section_number,
                                "title": title,
                                "subsections": [],
                                "estimated_hours": 1.5,
                                "status": "not_started",
                                "prerequisites": []
                            }
                            
                            # Process subsections
                            for subsection in node.get("sections", []):
                                subsection_data = build_roadmap(subsection, level + 1, chapter_number, section_number)
                                if subsection_data:
                                    section["subsections"].append(subsection_data)
                            
                            return section
                        
                        elif level == 2:  # Subsection level
                            return {
                                "subsection_id": f"ch_{chapter_number}_sec_{section_number}_sub_{len(node.get('sections', [])) + 1}",
                                "title": title,
                                "estimated_hours": 0.5,
                                "status": "not_started"
                            }
                
                return None
            
            build_roadmap(hierarchy)
            return roadmap
            
        except Exception as e:
            print(f"Error creating learning roadmap: {str(e)}")
            return {"chapters": [], "total_chapters": 0, "total_sections": 0, "estimated_learning_hours": 0}
    
    def get_concept_candidates(self, hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract potential concepts from hierarchical structure"""
        try:
            concepts = []
            
            def extract_from_node(node, level=0, parent_title=""):
                if "title" in node and node["title"]:
                    title = node["title"].strip()
                    concepts.append({
                        "title": title,
                        "level": level,
                        "parent": parent_title,
                        "content_preview": node.get("content", "")[:200],  # First 200 chars only
                        "type": "section"
                    })
                    current_parent = title
                else:
                    current_parent = parent_title
                
                for section in node.get("sections", []):
                    extract_from_node(section, level + 1, current_parent)
            
            extract_from_node(hierarchy)
            
            # Filter and rank concepts
            filtered_concepts = []
            for concept in concepts:
                title = concept["title"]
                if (len(title) > 3 and 
                    len(title) < 100 and 
                    not title.lower().startswith(('page', 'figure', 'table'))):
                    filtered_concepts.append(concept)
            
            return filtered_concepts[:20]  # Limit to top 20 concepts
            
        except Exception as e:
            print(f"Error extracting concept candidates: {str(e)}")
            return []

# Global parser instance
hierarchical_parser = HierarchicalDocumentParser()
