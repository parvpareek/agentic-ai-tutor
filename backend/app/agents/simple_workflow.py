# backend/app/agents/simple_workflow.py
"""
Core Agents for Adaptive AI Tutor

This file contains the fundamental agents for the learning system:
- IngestAgent: Document ingestion, parsing, and chunking
- ConceptExtractionAgent: Intelligent concept extraction from documents
- TutorEvaluatorAgent: Combined teaching and evaluation (Q&A, quizzes)
- SpacedRepetitionScheduler: Practice scheduling algorithm

Note: Planning and adaptation logic is handled by LLMPlannerAgent (see llm_planner.py)
which provides intelligent, agentic planning with remediation strategies.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
import asyncio
from datetime import datetime, timedelta
import json

# Simple state for the workflow
@dataclass
class SimpleState:
    """Simplified state for the 4-agent workflow"""
    # Document ingestion
    doc_id: Optional[str] = None
    session_id: Optional[str] = None
    ingest_status: str = "pending"  # pending, processing, completed, failed
    concepts: List[Dict] = None  # [{concept_id, label, supporting_chunk_ids[]}]
    
    # Student interaction
    student_choice: Optional[str] = None  # "diagnostic", "choose_topics", "from_beginning"
    selected_topics: List[str] = None
    
    # Planning
    study_plan: List[Dict] = None  # [{step_id, action, topic, est_minutes, why_assigned}]
    current_step: int = 0
    current_topic: Optional[str] = None  # Track current topic being studied
    learning_roadmap: Optional[Dict] = None  # Store learning roadmap for reference
    
    # Tutoring
    current_question: Optional[str] = None
    current_quiz: Optional[Dict] = None
    student_answers: List[str] = None
    taught_content: Dict[str, str] = None  # {topic: explanation} - Store what was actually taught
    
    # Evaluation
    student_profile: Dict[str, Any] = None  # {topic: {proficiency, attempts, last_practiced}}
    
    # Spaced repetition
    practice_schedule: Dict[str, datetime] = None  # {topic: next_practice_date}

class AgentType(Enum):
    INGEST = "ingest"
    CONCEPT_EXTRACTION = "concept_extraction"
    PLANNER = "planner"
    TUTOR_EVALUATOR = "tutor_evaluator"

@dataclass
class AgentResponse:
    success: bool
    data: Dict[str, Any]
    reasoning: str
    next_agent: Optional[AgentType] = None

class IngestAgent:
    """Agent 1: Extract text from uploaded files, chunk, index embeddings
    
    🌟 MULTI-DOCUMENT SUPPORT:
    - Processes multiple documents robustly
    - Creates unified outline across all documents
    - Maintains document source tracking for grounding
    """
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.name = "IngestAgent"
        # Import hierarchical parser
        from app.core.hierarchical_parser import hierarchical_parser
        self.hierarchical_parser = hierarchical_parser
    
    async def execute(self, file_content, filename: str) -> AgentResponse:
        """Process single uploaded file - wrapper for multi-document support"""
        return await self.execute_multi([file_content], [filename])
    
    async def execute_multi(self, file_contents: List, filenames: List[str]) -> AgentResponse:
        """
        🌟 Process MULTIPLE uploaded files and create unified learning structure
        
        KEY FEATURES:
        - Unified document outline across all files
        - Document source tracking for grounding
        - Intelligent concept extraction context
        """
        print(f"[{self.name}] 📚 Processing {len(filenames)} document(s): {', '.join(filenames)}")
        
        try:
            all_chunks = []
            all_outlines = []
            all_roadmaps = []
            document_registry = []  # Track all documents for grounding
            
            # Process each document
            for idx, (file_content, filename) in enumerate(zip(file_contents, filenames)):
                doc_id = f"doc_{uuid.uuid4().hex[:8]}"
                print(f"\n[{self.name}] === Document {idx+1}/{len(filenames)}: {filename} ===")
                
                # Parse document hierarchically
                if isinstance(file_content, str):
                    content_bytes = file_content.encode('utf-8')
                else:
                    content_bytes = file_content
                
                parse_result = self.hierarchical_parser.parse_document(
                    content_bytes, 
                    filename
                )
                
                # Extract document outline and roadmap
                if not parse_result.get("success", False):
                    print(f"[{self.name}] Doc {idx+1} parsing failed, using simple chunking")
                    chunks = self._chunk_content(file_content, doc_id, doc_source=filename)
                    doc_outline = f"📄 Document {idx+1}: {filename} (structure unavailable)"
                    doc_roadmap = {"chapters": [], "total_chapters": 0, "total_sections": 0}
                else:
                    # Extract outline, roadmap and hierarchy
                    hierarchy = parse_result.get("hierarchy", {})
                    doc_outline = self.hierarchical_parser.get_document_outline(hierarchy)
                    doc_roadmap = self.hierarchical_parser.get_learning_roadmap(hierarchy)
                    
                    # Add document source prefix to outline
                    doc_outline = f"📄 === DOCUMENT {idx+1}: {filename} ===\n{doc_outline}"
                    
                    print(f"[{self.name}] Doc {idx+1}: {doc_roadmap['total_chapters']} chapters, {doc_roadmap['total_sections']} sections")
                    
                    # Create chunks with document source tracking
                    raw_chunks = parse_result.get("raw_chunks", [])
                    if raw_chunks and len(raw_chunks) > 5:
                        print(f"[{self.name}] Doc {idx+1}: Using {len(raw_chunks)} raw chunks")
                        chunks = self._create_chunks_from_raw(raw_chunks, doc_id, doc_source=filename)
                    else:
                        print(f"[{self.name}] Doc {idx+1}: Using hierarchical chunking")
                        chunks = self._create_hierarchical_chunks(parse_result, doc_id, doc_source=filename)
                
                print(f"[{self.name}] Doc {idx+1}: Created {len(chunks)} chunks")
                
                # Accumulate
                all_chunks.extend(chunks)
                all_outlines.append(doc_outline)
                all_roadmaps.append(doc_roadmap)
                document_registry.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_count": len(chunks),
                    "chapters": doc_roadmap.get("total_chapters", 0),
                    "sections": doc_roadmap.get("total_sections", 0)
                })
            
            # Store all chunks in vectorstore
            print(f"\n[{self.name}] 💾 Storing {len(all_chunks)} total chunks across {len(filenames)} documents")
            self.vectorstore.upsert_chunks(all_chunks)
            
            # Create unified outline
            unified_outline = self._create_unified_outline(all_outlines, filenames)
            
            # Create unified roadmap
            unified_roadmap = self._merge_roadmaps(all_roadmaps)
            
            print(f"[{self.name}] ✅ Multi-document ingestion complete!")
            print(f"[{self.name}] 📊 Total: {len(all_chunks)} chunks, {unified_roadmap['total_chapters']} chapters, {unified_roadmap['total_sections']} sections")
            
            return AgentResponse(
                success=True,
                data={
                    "doc_id": "multi_doc",  # Unified identifier
                    "ingest_status": "completed",
                    "chunks_created": len(all_chunks),
                    "document_outline": unified_outline,
                    "learning_roadmap": unified_roadmap,
                    "document_registry": document_registry,  # For grounding
                    "document_count": len(filenames)
                },
                reasoning=f"Successfully ingested {len(filenames)} documents into {len(all_chunks)} chunks with unified roadmap",
                next_agent=AgentType.CONCEPT_EXTRACTION
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={"ingest_status": "failed"},
                reasoning=f"Ingest failed: {str(e)}"
            )
    
    def _create_unified_outline(self, outlines: List[str], filenames: List[str]) -> str:
        """Create unified outline from multiple documents"""
        if len(outlines) == 1:
            return outlines[0]
        
        unified = "📚 UNIFIED DOCUMENT OUTLINE (Multiple Documents):\n\n"
        unified += f"Total Documents: {len(filenames)}\n"
        unified += f"Documents: {', '.join(filenames)}\n\n"
        unified += "=" * 80 + "\n\n"
        
        for outline in outlines:
            unified += outline + "\n\n"
            unified += "=" * 80 + "\n\n"
        
        return unified
    
    def _merge_roadmaps(self, roadmaps: List[Dict]) -> Dict:
        """Merge multiple learning roadmaps into one"""
        merged = {
            "chapters": [],
            "total_chapters": 0,
            "total_sections": 0,
            "estimated_learning_hours": 0,
            "document_count": len(roadmaps)
        }
        
        for idx, roadmap in enumerate(roadmaps):
            merged["total_chapters"] += roadmap.get("total_chapters", 0)
            merged["total_sections"] += roadmap.get("total_sections", 0)
            merged["estimated_learning_hours"] += roadmap.get("estimated_learning_hours", 0)
            
            # Add chapters with document prefix
            for chapter in roadmap.get("chapters", []):
                chapter_copy = chapter.copy()
                chapter_copy["document_index"] = idx
                merged["chapters"].append(chapter_copy)
        
        return merged
    
    def _chunk_content(self, content: str, doc_id: str, doc_source: str = "unknown") -> List[Dict]:
        """Chunk content into smaller pieces for embedding with document source tracking"""
        chunk_size = 500
        overlap = 50
        chunks = []
        
        chunk_index = 0
        for i in range(0, len(content), chunk_size - overlap):
            chunk_text = content[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "id": f"{doc_id}_chunk_{chunk_index}",
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "source": "uploaded_file",
                        "doc_source": doc_source  # 🌟 Document source for grounding
                    }
                })
                chunk_index += 1
        
        return chunks
    
    def _create_chunks_from_raw(self, raw_chunks: List[Dict], doc_id: str, doc_source: str = "unknown") -> List[Dict]:
        """Create indexed chunks from raw llmsherpa chunks (SIMPLE & EFFECTIVE).
        These chunks have actual content, not just headings!
        """
        chunks = []
        for idx, raw_chunk in enumerate(raw_chunks):
            text = raw_chunk.get("text", "")
            section_title = raw_chunk.get("section_title", "Document")
            
            # Skip if no content
            if not text or len(text.strip()) < 50:
                continue
            
            # Create chunk with section context
            chunk_text = f"{section_title}\n\n{text}"
            
            chunks.append({
                "id": f"{doc_id}_raw_{idx}",
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "section_title": section_title,
                    "page_idx": raw_chunk.get("page_idx", 0),
                    "source": "raw_chunks",
                    "doc_source": doc_source  # 🌟 Document source for grounding
                }
            })
        
        return chunks
    
    def _create_hierarchical_chunks(self, parse_result: Dict, doc_id: str, doc_source: str = "unknown") -> List[Dict]:
        """Create chunks from hierarchical document structure with actual content.
        Applies sanitization and skips heading-only/garbage content to improve retrieval quality.
        """
        try:
            chunks = []
            chunk_index = 0
            chunk_size = 600  # Target chunk size
            
            # Extract sections from parse result
            sections = parse_result.get("sections", [])
            
            print(f"[{self.name}] Processing {len(sections)} sections for chunking")
            
            # Helper to sanitize text: remove non-printable and collapse whitespace
            import re, string
            printable = set(string.printable)
            def sanitize_text(txt: str) -> str:
                if not txt:
                    return ""
                # Remove non-printable
                txt = ''.join(ch for ch in txt if ch in printable)
                # Normalize whitespace
                txt = re.sub(r"\s+", " ", txt).strip()
                return txt

            # Create chunks from sections with actual content
            for section in sections:
                title = section.get("title", "")
                content = section.get("content", "")
                level = section.get("level", 1)
                
                # Sanitize
                stitle = sanitize_text(title)
                scontent = sanitize_text(content)

                # Skip if no meaningful content or content equals title
                if not scontent or len(scontent) < 80 or scontent.lower() == stitle.lower():
                    continue
                
                # For large sections, split into multiple chunks while preserving context
                if len(scontent) > chunk_size:
                    # Split by paragraphs
                    paragraphs = re.split(r"\n\s*\n", scontent)
                    current_chunk = f"{stitle}\n\n"
                    
                    for para in paragraphs:
                        if not para.strip():
                            continue
                        
                        # If adding this paragraph exceeds chunk_size, save current chunk
                        if len(current_chunk) + len(para) > chunk_size and len(current_chunk) > len(f"{stitle}\n\n"):
                            chunks.append({
                                "id": f"{doc_id}_section_{chunk_index}",
                                "text": current_chunk.strip(),
                                "metadata": {
                                    "doc_id": doc_id,
                                    "chunk_index": chunk_index,
                                    "source": "hierarchical_section",
                                    "section_title": stitle,
                                    "section_level": level,
                                    "document_type": parse_result.get("document_type", "unknown"),
                                    "doc_source": doc_source  # 🌟 Document source for grounding
                                }
                            })
                            chunk_index += 1
                            # Start new chunk with section context
                            current_chunk = f"{stitle}\n\n{para}\n\n"
                        else:
                            current_chunk += para + "\n\n"
                    
                    # Add remaining content
                    if len(current_chunk.strip()) > len(f"{stitle}"):
                        chunks.append({
                            "id": f"{doc_id}_section_{chunk_index}",
                            "text": current_chunk.strip(),
                            "metadata": {
                                "doc_id": doc_id,
                                "chunk_index": chunk_index,
                                "source": "hierarchical_section",
                                "section_title": stitle,
                                "section_level": level,
                                "document_type": parse_result.get("document_type", "unknown")
                            }
                        })
                        chunk_index += 1
                else:
                    # Section is small enough to be one chunk
                    chunk_text = f"{stitle}\n\n{scontent}"
                    chunks.append({
                        "id": f"{doc_id}_section_{chunk_index}",
                        "text": chunk_text,
                        "metadata": {
                            "doc_id": doc_id,
                            "chunk_index": chunk_index,
                            "source": "hierarchical_section",
                            "section_title": stitle,
                            "section_level": level,
                            "document_type": parse_result.get("document_type", "unknown")
                        }
                    })
                    chunk_index += 1
            
            print(f"[{self.name}] Created {len(chunks)} hierarchical chunks from sections")
            
            # If no meaningful chunks from sections, use full text chunking
            if len(chunks) < 3:
                print(f"[{self.name}] Too few hierarchical chunks, falling back to full text chunking")
                full_text = parse_result.get("full_text", "")
                if full_text:
                    chunks = self._chunk_content(full_text, doc_id)
            
            return chunks
            
        except Exception as e:
            print(f"[{self.name}] Error creating hierarchical chunks: {str(e)}")
            import traceback
            print(f"[{self.name}] Traceback: {traceback.format_exc()}")
            # Fallback to simple chunking
            full_text = parse_result.get("full_text", "")
            return self._chunk_content(full_text, doc_id)

class ConceptExtractionAgent:
    """Agent 2: Extract concepts/topics from indexed content using document outline
    
    🌟 MULTI-DOCUMENT SUPPORT:
    - Extracts concepts intelligently across multiple documents
    - Aligns concepts with document structure for easier teaching
    - Maintains document source tracking for grounding
    """
    
    def __init__(self, vectorstore, llm, memory=None):
        self.vectorstore = vectorstore
        self.llm = llm
        self.memory = memory  # Redis memory for storing segment plans
        self.name = "ConceptExtractionAgent"
    
    async def execute(self, doc_id: str, document_outline: str = None, 
                     learning_roadmap: Dict = None, target_exam: str = "JEE", 
                     student_context: str = None, document_registry: List[Dict] = None) -> AgentResponse:
        """
        🌟 Extract key concepts from document(s) using unified outline and registry.
        
        MULTI-DOCUMENT INTELLIGENCE:
        - Uses unified outline across all documents
        - Extracts concepts that align with document structure
        - Maintains source tracking for grounding
        """
        doc_count = learning_roadmap.get("document_count", 1) if learning_roadmap else 1
        print(f"[{self.name}] 📚 Extracting concepts from {doc_count} document(s)")
        
        try:
            # If we have a document outline and roadmap, use intelligent extraction
            if (document_outline and document_outline != "Document structure not available" and
                learning_roadmap and learning_roadmap.get("chapters")):
                print(f"[{self.name}] Using deep structure-based intelligent extraction")
                concepts = await self._extract_concepts_from_deep_structure(
                    document_outline, learning_roadmap, target_exam, student_context
                )
            elif document_outline and document_outline != "Document structure not available":
                print(f"[{self.name}] Using outline-based extraction")
                concepts = await self._extract_concepts_from_outline(
                    document_outline, target_exam, student_context
                )
            else:
                # No outline available - cannot extract concepts reliably
                print(f"[{self.name}] ERROR: No document outline available for concept extraction")
                return AgentResponse(
                    success=False,
                    data={},
                    reasoning="Document parsing failed - no outline available for concept extraction. Please ensure document is properly formatted."
                )
            
            print(f"[{self.name}] Extracted {len(concepts)} concepts")
            
            # Store segment plans in Redis memory if available
            if self.memory:
                for concept in concepts:
                    topic = concept.get("label", "")
                    segments = concept.get("learning_segments", [])
                    if segments:
                        success = self.memory.store_segment_plan(topic, segments)
                        if success:
                            print(f"[{self.name}] Stored {len(segments)} segments for topic: {topic}")
                        else:
                            print(f"[{self.name}] Failed to store segments for topic: {topic}")
            
            return AgentResponse(
                success=True,
                data={"concepts": concepts, "learning_roadmap": learning_roadmap},
                reasoning=f"Successfully extracted {len(concepts)} relevant concepts",
                next_agent=AgentType.PLANNER
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Concept extraction failed: {str(e)}"
            )
    
    async def _extract_concepts_from_outline(self, outline: str, target_exam: str, 
                                            student_context: str = None) -> List[Dict]:
        """
        Intelligently extract concepts from document outline using LLM.
        LLM corrects any parsing errors and identifies truly relevant learning topics.
        """
        try:
            context_str = f"\nStudent Context: {student_context}" if student_context else ""
            
            prompt = f"""You are an expert educational content analyzer.

MULTI-DOCUMENT CONCEPT EXTRACTION

Analyze this document outline (may contain multiple documents) and extract KEY LEARNING CONCEPTS for {target_exam} preparation.

DOCUMENT OUTLINE:
{outline[:3000]}

CRITICAL INSTRUCTIONS - ALIGN WITH DOCUMENT STRUCTURE:
1. **Follow Document Organization**: Extract concepts that MATCH the document's structure
   - If document has "Chapter 1: Intro, Chapter 2: Advanced Topics", your concepts should reflect this
   - Segment titles should ALIGN with section/subsection titles in the outline
   - This makes teaching easier: retrieve content directly from matching sections!

2. **Why Alignment Matters**:
   - Student uploads a textbook organized by topics -> extract those same topics
   - When teaching "Tokenization", we'll retrieve chunks from "Tokenization" section -> perfect grounding!
   - Without alignment: we invent segments that don't match document -> poor retrieval, hallucinations

3. **Multi-Document Intelligence**:
   - If multiple documents (DOCUMENT 1, DOCUMENT 2), extract concepts from each
   - Concepts can span documents if topics are related
   - Maintain logical flow across documents

OUTPUT FORMAT - Return ONLY valid JSON array, no code fences, no extra text:
[
  {{
    "concept_id": "unique_id",
    "label": "Concept Name (2-4 words)",
    "description": "Brief 1-sentence description",
    "section_title": "EXACT section from outline",
    "document_source": "filename if multi-document",
    "difficulty": "beginner|intermediate|advanced",
    "search_queries": ["query1", "query2", "query3"],
    "learning_segments": [
      {{
        "segment_id": "seg_1",
        "title": "EXACT subsection title from outline",
        "order": 1,
        "estimated_minutes": 7,
        "learning_objectives": ["objective"],
        "prerequisites": [],
        "source_section": "Exact section path from outline"
      }}
    ]
  }}
]

RULES:
- Extract 5-12 distinct concepts from the outline
- Each concept has 3-5 learning segments (NOT more)
- **SEGMENT TITLES MUST MATCH DOCUMENT SECTIONS** (e.g., if outline says "1.2 Word Embeddings", segment should be "Word Embeddings")
- Ignore noise: "Introduction", "Conclusion", page numbers
- Focus on teachable, testable topics that exist in the documents
- Return ONLY the JSON array, nothing else

EXAMPLE (good alignment):
Document outline: "1. Text Processing -> 1.1 Tokenization -> 1.2 Normalization"
Your segments: [{{"title": "Tokenization", ...}}, {{"title": "Normalization", ...}}]
Perfect alignment -> easy retrieval!"""
            
            response = await self.llm.ainvoke(prompt)
            
            # Robust JSON array parsing (strip code fences, extract first array)
            import json
            import re
            
            # Debug: Log raw response
            raw_content = response.content if response else ""
            print(f"[{self.name}] LLM raw response length: {len(raw_content)} chars")
            if len(raw_content) < 200:
                print(f"[{self.name}] LLM raw response (full): {raw_content}")
            else:
                print(f"[{self.name}] LLM raw response (first 500 chars): {raw_content[:500]}")
            
            content = (raw_content or "").strip()
            
            if not content:
                print(f"[{self.name}] ERROR: LLM returned empty response")
                raise ValueError("LLM returned empty response")
            
            # Remove code fences if present
            if content.startswith('```'):
                if '```json' in content:
                    # Remove opening fence: ```json
                    content = content.split('```json', 1)[1]
                    # Remove closing fence: ```
                    if '```' in content:
                        content = content.split('```')[0]
                else:
                    # Generic fence removal
                    parts = content.split('```')
                    if len(parts) >= 2:
                        content = parts[1]
                content = content.strip()
            
            # Extract JSON array if wrapped in text
            if not content.startswith('['):
                m = re.search(r"\[[\s\S]*\]", content)
                if m:
                    content = m.group(0)
                else:
                    print(f"[{self.name}] ERROR: No JSON array found in response")
                    print(f"[{self.name}] Content after cleanup: {content[:500]}")
                    raise ValueError("No JSON array found in LLM response")
            
            # Parse JSON
            try:
                concepts = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[{self.name}] JSON parse error: {e}")
                print(f"[{self.name}] Content being parsed (first 1000 chars): {content[:1000]}")
                raise
            
            if not isinstance(concepts, list):
                raise ValueError(f"Concepts JSON is not a list, got: {type(concepts)}")
            
            print(f"[{self.name}] Successfully parsed {len(concepts)} concepts")
            return concepts[:12]
                
        except Exception as e:
            print(f"[{self.name}] Error in outline-based extraction: {str(e)}")
            return []
    
    async def _extract_concepts_from_deep_structure(self, outline: str, roadmap: Dict, 
                                                   target_exam: str, student_context: str = None) -> List[Dict]:
        """
        Extract concepts using deep document structure (outline + roadmap).
        This gives the LLM maximum context about document organization and learning progression.
        """
        try:
            context_str = f"\nStudent Context: {student_context}" if student_context else ""
            
            # Format roadmap for LLM
            roadmap_str = f"""
LEARNING ROADMAP:
Total Chapters: {roadmap.get('total_chapters', 0)}
Total Sections: {roadmap.get('total_sections', 0)}
Estimated Learning Hours: {roadmap.get('estimated_learning_hours', 0)}

Chapter Structure:
"""
            for chapter in roadmap.get('chapters', [])[:5]:  # Limit to first 5 chapters
                roadmap_str += f"\nChapter {chapter.get('chapter_number', '?')}: {chapter.get('title', 'Unknown')}"
                roadmap_str += f" (Est. {chapter.get('estimated_hours', 0)} hours)"
                
                for section in chapter.get('sections', [])[:3]:  # First 3 sections per chapter
                    roadmap_str += f"\n  - Section {section.get('section_number', '?')}: {section.get('title', 'Unknown')}"
                    for subsection in section.get('subsections', [])[:2]:  # First 2 subsections
                        roadmap_str += f"\n    * {subsection.get('title', 'Unknown')}"
            
            # Detect multi-document scenario
            doc_count = roadmap.get('document_count', 1)
            multi_doc_note = ""
            if doc_count > 1:
                multi_doc_note = f"\nNOTE: This is a UNIFIED outline from {doc_count} documents. Extract concepts across all documents while maintaining logical flow."
            
            prompt = f"""MULTI-DOCUMENT CONCEPT EXTRACTION

Analyze this document structure and extract KEY LEARNING CONCEPTS for {target_exam}.

DOCUMENT STRUCTURE:
{outline[:2000]}

{roadmap_str[:800]}{multi_doc_note}

CRITICAL - ALIGN SEGMENTS WITH DOCUMENT STRUCTURE:
- Segment titles MUST match section/subsection titles from the outline above
- If outline says "2.3 Word Embeddings", your segment should be "Word Embeddings"
- If outline says "Chapter 4: Neural Networks -> 4.1 Perceptrons", segment should be "Perceptrons"
- WHY? When teaching, we retrieve content using segment title as search query
  -> Aligned titles = perfect retrieval = grounded teaching!

Return ONLY a JSON array (no code fences):
[
  {{
    "concept_id": "id", 
    "label": "Name (2-4 words)", 
    "description": "Brief description",
    "chapter": "Chapter X",
    "section": "Section Y",
    "document_source": "filename if multi-document",
    "difficulty": "beginner|intermediate|advanced",
    "search_queries": ["query1", "query2"],
    "learning_segments": [
      {{
        "segment_id": "seg_1",
        "title": "EXACT section/subsection title from outline",
        "order": 1,
        "estimated_minutes": 7,
        "learning_objectives": ["obj"],
        "prerequisites": [],
        "source_section": "Full section path from outline"
      }}
    ]
  }}
]

RULES:
- Extract 10-12 concepts
- Each has 3-5 segments (NOT more)
- **SEGMENT TITLES = DOCUMENT SECTION TITLES** (this is critical for retrieval!)
- Focus on testable topics that exist in documents
- Ignore intro/conclusion/references
- Maintain logical progression

EXAMPLE (good alignment):
Outline: "Chapter 3: NLP Basics -> 3.1 Tokenization -> 3.2 Stemming -> 3.3 Lemmatization"
Your segments: [{{"title": "Tokenization"}}, {{"title": "Stemming"}}, {{"title": "Lemmatization"}}]
Perfect! Easy retrieval!"""
            
            response = await self.llm.ainvoke(prompt)
            
            # Robust JSON array parsing
            import json
            import re
            
            # Debug: Log raw response
            raw_content = response.content if response else ""
            print(f"[{self.name}] LLM raw response length: {len(raw_content)} chars")
            if len(raw_content) < 200:
                print(f"[{self.name}] LLM raw response (full): {raw_content}")
            else:
                print(f"[{self.name}] LLM raw response (first 500 chars): {raw_content[:500]}")
            
            content = (raw_content or "").strip()
            
            if not content:
                print(f"[{self.name}] ERROR: LLM returned empty response")
                raise ValueError("LLM returned empty response")
            
            # Remove code fences if present
            if content.startswith('```'):
                if '```json' in content:
                    # Remove opening fence: ```json
                    content = content.split('```json', 1)[1]
                    # Remove closing fence: ```
                    if '```' in content:
                        content = content.split('```')[0]
                else:
                    # Generic fence removal
                    parts = content.split('```')
                    if len(parts) >= 2:
                        content = parts[1]
                content = content.strip()
            
            # Extract JSON array if wrapped in text
            if not content.startswith('['):
                m = re.search(r"\[[\s\S]*\]", content)
                if m:
                    content = m.group(0)
                else:
                    print(f"[{self.name}] ERROR: No JSON array found in response")
                    print(f"[{self.name}] Content after cleanup: {content[:500]}")
                    raise ValueError("No JSON array found in LLM response")
            
            # Parse JSON
            try:
                concepts = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[{self.name}] JSON parse error: {e}")
                print(f"[{self.name}] Content being parsed (first 1000 chars): {content[:1000]}")
                raise
            
            if not isinstance(concepts, list):
                raise ValueError(f"Concepts JSON is not a list, got: {type(concepts)}")
            
            print(f"[{self.name}] Successfully parsed {len(concepts)} concepts")
            return concepts[:15]
                
        except Exception as e:
            print(f"[{self.name}] Error in deep structure extraction: {str(e)}")
            return []

class TutorEvaluatorAgent:
    """Agent 4: Combined tutor and evaluator for RAG answers and quiz generation"""
    
    def __init__(self, vectorstore, llm, database):
        self.vectorstore = vectorstore
        self.llm = llm
        self.db = database
        self.name = "TutorEvaluatorAgent"
    
    async def execute(self, action: str, **kwargs) -> AgentResponse:
        """Execute tutoring or evaluation action"""
        print(f"[{self.name}] Executing action: {action}")
        
        try:
            if action == "answer_question":
                return await self._answer_question(**kwargs)
            elif action == "generate_quiz":
                return await self._generate_quiz(**kwargs)
            elif action == "evaluate_quiz":
                return await self._evaluate_quiz(**kwargs)
            elif action == "update_profile":
                return await self._update_profile(**kwargs)
            else:
                return AgentResponse(
                    success=False,
                    data={},
                    reasoning=f"Unknown action: {action}"
                )
                
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Tutor/Evaluator failed: {str(e)}"
            )
    
    async def _answer_question(self, question: str, topic: str = None, recent_summaries: Optional[List[Dict]] = None, recent_qa: Optional[List[Dict]] = None, current_segment_id: Optional[str] = None, current_segment_text: str = "") -> AgentResponse:
        """ENHANCED: Provide adaptive RAG-based answer using confusion pattern analysis"""
        print(f"[{self.name}] Answering question: {question}")
        
        try:
            # ENHANCEMENT: Detect confusion patterns
            from app.core.learning_patterns import LearningPatternAnalyzer
            from app.core.agent_thoughts import thoughts_tracker
            
            analyzer = LearningPatternAnalyzer()
            
            confusion_analysis = analyzer.detect_confusion_type(question, recent_qa or [])
            confusion_type = confusion_analysis["confusion_type"]
            is_repeat = confusion_analysis["is_repeat"]
            approach = confusion_analysis["suggested_approach"]
            
            # Log agent thinking
            thought = f"Detected {confusion_type} confusion. {'Repeat question - ' if is_repeat else ''}Strategy: {approach}"
            thoughts_tracker.add("TutorEvaluator", thought, "🧠", {
                "confusion_type": confusion_type,
                "is_repeat": is_repeat,
                "approach": approach
            })
            
            print(f"[{self.name}] 🧠 Confusion: {confusion_type}, Repeat: {is_repeat}, Approach: {approach}")
            
            # Retrieve relevant content with enhanced query
            seg_hint = f" {current_segment_id}" if current_segment_id else ""
            # Add confusion keywords to query
            keywords = " ".join(confusion_analysis["keywords"])
            query = f"{question} {topic or ''}{seg_hint} {keywords}"
            results = self.vectorstore.query_top_k(query, k=4)  # More chunks for confused students
            
            documents = results.get('documents', [[]])[0] if 'documents' in results else []
            sources = results.get('metadatas', [[]])[0] if 'metadatas' in results else []
            
            # Build context with priority to current segment
            context_parts = []
            if current_segment_text:
                context_parts.append(current_segment_text[:2000])
            if documents:
                context_parts.append("\n\n".join(documents[:3]))  # Top 3 chunks
            context = "\n\n".join([p for p in context_parts if p.strip()]) or "No relevant content found."
            summaries = recent_summaries or []
            
            # Build Q&A history
            qa_history_str = ""
            if recent_qa:
                qa_history_str = "\nRECENT CONVERSATION:\n"
                for i, qa in enumerate(recent_qa[:3]):
                    qa_history_str += f"{i+1}. Student: {qa.get('q', '')}\n   You: {qa.get('a', '')[:150]}...\n"
            
            # ADAPTIVE PROMPT based on confusion type
            confusion_specific_instruction = {
                "concept": """
                STRATEGY: Definitional Clarity
                - Start with a ONE-SENTENCE clear definition
                - Follow with a concrete, relatable example
                - Contrast with what it's NOT (common misconception)
                - End with why it matters for their learning""",
                
                "application": """
                STRATEGY: Step-by-Step Guidance  
                - Break down the process into numbered steps
                - Provide a worked example showing each step
                - Highlight common mistakes to avoid
                - Give them a similar problem to try""",
                
                "connection": """
                STRATEGY: Relationship Building
                - Explain how concepts relate using analogies
                - Draw explicit connections to what they already know
                - Use "bridge" concepts to link ideas
                - Create a mini concept map in text""",
                
                "vague": """
                STRATEGY: Comprehensive Recap
                - Assume confusion about current segment
                - Provide fresh explanation with different angle
                - Use multiple examples and analogies
                - Check for specific confusion points"""
            }
            
            repeat_warning = """
            ⚠️ CRITICAL: This is a REPEAT question - previous answer didn't work!
            - Try a COMPLETELY DIFFERENT approach/analogy
            - Use simpler language and more visual descriptions
            - Break down into smaller pieces
            - Ask clarifying questions to pinpoint confusion""" if is_repeat else ""
            
            prompt = f"""You are an adaptive tutor helping a student with "{current_segment_id or topic}".

STUDENT'S QUESTION: "{question}"

{repeat_warning}

CONFUSION ANALYSIS:
- Type: {confusion_type}
- Recommended Approach: {approach}

{confusion_specific_instruction.get(confusion_type, "")}

CONTEXT FROM SEGMENT:
            {context[:2500]}
            
PRIOR LEARNING:
{json.dumps(summaries)[:800] if summaries else "First segment"}
            {qa_history_str}
            
RESPONSE GUIDELINES:
1. Address the {confusion_type} confusion directly
2. Use the recommended strategy above
3. Keep answer focused and structured (use headings, bullets)
4. Include ONE concrete example that builds on their prior knowledge
5. End with a quick comprehension checkpoint question
6. Use encouraging, supportive tone

Make your answer clear, actionable, and adapted to their confusion type!"""
            
            # Call LLM and log prompt/response to session log file
            import time
            from app.utils.session_logger import get_logger
            start = time.time()
            response = await self.llm.ainvoke(prompt)
            duration = time.time() - start
            try:
                get_logger().log_llm_call(self.name, "answer_question", prompt, response.content, duration)
            except Exception:
                pass
            
            return AgentResponse(
                success=True,
                data={
                    "answer": response.content,
                    "sources": [meta.get('source', 'unknown') for meta in sources],
                    "supporting_chunks": [meta.get('chunk_id', 'unknown') for meta in sources]
                },
                reasoning="Generated RAG-based answer with sources"
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Answer generation failed: {str(e)}"
            )
    
    async def _generate_quiz(self, topic: str, concept_id: str, difficulty: str = "medium", num_questions: int = 5, taught_content: str = "", segment_id: str = None, exam_context: Dict = None, student_context: Dict = None, include_subjective: bool = False) -> AgentResponse:
        """Generate quiz questions for a topic based on what was taught and student's learning context"""
        print(f"[{self.name}] Generating {num_questions} {difficulty} questions for {topic} (subjective={include_subjective})")
        print(f"[{self.name}] Using {'taught content' if taught_content else 'retrieved context'} for grounding")
        
        from app.core.agent_thoughts import thoughts_tracker
        
        # Parse student context for targeted questions
        student_context = student_context or {}
        student_questions = student_context.get("recent_questions", [])
        unclear_segments = student_context.get("unclear_segments", [])
        confusion_areas = student_context.get("confusion_areas", [])
        
        # Log quiz generation strategy
        context_sources = []
        if taught_content:
            context_sources.append(f"{len(taught_content)} chars of taught content")
        if student_questions:
            context_sources.append(f"{len(student_questions)} student questions")
        if unclear_segments:
            context_sources.append(f"{len(unclear_segments)} unclear segments")
        
        if context_sources:
            thoughts_tracker.add(
                "Tutor",
                f"Generating {num_questions} {difficulty} quiz questions using: {', '.join(context_sources)}",
                "📝",
                {"difficulty": difficulty, "subjective": include_subjective, "topic": topic}
            )
        
        if student_questions or unclear_segments or confusion_areas:
            print(f"[{self.name}] Using student context: {len(student_questions)} questions, {len(unclear_segments)} unclear, {len(confusion_areas)} confusions")
        
        # Parse exam context for question style
        exam_type = "general"
        exam_style_instruction = ""
        if exam_context:
            exam_type = exam_context.get("exam_type", "general")
            
            if exam_type == "JEE":
                exam_style_instruction = """
**EXAM STYLE: JEE (Technical & Numerical)**
- Questions should test deep conceptual understanding AND calculation ability
- Include numerical problem-solving questions where applicable
- Use technical terminology and precise language
- Options should include common calculation errors as distractors
- Explanations should show mathematical steps when relevant"""
            elif exam_type == "UPSC":
                exam_style_instruction = """
**EXAM STYLE: UPSC (Comprehensive & Analytical)**
- Questions should test broad understanding and analytical thinking
- Focus on conceptual clarity, real-world applications, and interconnections
- Use clear, articulate language suitable for descriptive answers
- Options should test nuanced understanding
- Explanations should provide comprehensive context"""
            elif exam_type in ["SAT", "GRE"]:
                exam_style_instruction = f"""
**EXAM STYLE: {exam_type} (Standardized Test)**
- Questions should test reasoning and problem-solving skills
- Balance between conceptual and applied knowledge
- Use clear, unambiguous language
- Options should be plausible and test critical thinking
- Explanations should clarify the reasoning process"""
            
        print(f"[{self.name}] Quiz tailored for exam type: {exam_type}")
        
        try:
            # CRITICAL FIX: Use ALL taught content, distributed across segments
            if taught_content:
                # Split taught content into segments and sample from each
                segments = taught_content.split("\n\n---\n\n")
                print(f"[{self.name}] Found {len(segments)} taught segments, total {len(taught_content)} chars")
                
                # Take balanced samples from each segment (not just the last one!)
                if len(segments) > 1:
                    chars_per_segment = 800  # ~800 chars per segment
                    context_parts = []
                    for i, seg in enumerate(segments):
                        if seg.strip():
                            context_parts.append(f"[Segment {i+1}] {seg[:chars_per_segment]}")
                    context = "\n\n".join(context_parts)
                else:
                    context = taught_content[:3000]  # Single segment, use more
                
                print(f"[{self.name}] Using balanced content ({len(context)} chars) from all segments for quiz")
            else:
                # Fallback: retrieve from vector store
                queries = [
                    topic,
                    f"{topic} definition examples",
                    f"{topic} key concepts important points"
                ]
                
                all_documents = []
                seen_texts = set()
                
                for query in queries:
                    results = self.vectorstore.query_top_k(query, k=4)
                    docs = results.get('documents', [[]])[0] if 'documents' in results else []
                    for doc in docs:
                        if doc not in seen_texts and len(doc.strip()) > 100:
                            seen_texts.add(doc)
                            all_documents.append(doc)
                
                documents = all_documents[:5]  # Top 5 unique chunks
                context = "\n\n---\n\n".join(documents) if documents else f"Topic: {topic}"
                print(f"[{self.name}] Retrieved {len(documents)} unique chunks for quiz generation")
            
            # Build student context section for targeted questions
            student_context_section = ""
            if student_questions or unclear_segments or confusion_areas:
                context_parts = []
                if student_questions:
                    context_parts.append(f"**Student's Recent Questions/Doubts:**\n" + "\n".join([f"- {q}" for q in student_questions[:3]]))
                if unclear_segments:
                    context_parts.append(f"**Segments Student Found Unclear:**\n" + "\n".join([f"- {s}" for s in unclear_segments]))
                if confusion_areas:
                    context_parts.append(f"**Areas of Confusion:**\n" + "\n".join([f"- {c}" for c in confusion_areas]))
                
                student_context_section = f"""
**STUDENT LEARNING CONTEXT** (create questions targeting these areas):
{chr(10).join(context_parts)}

IMPORTANT: Prioritize questions that address the student's questions, doubts, and unclear areas above.
At least {min(3, num_questions)} questions should directly relate to their learning context.
"""
            
            # Build question type instruction based on include_subjective
            question_type_instruction = ""
            if include_subjective:
                question_type_instruction = f"""
**QUESTION MIX** (for {num_questions} questions):
- {num_questions - 2} Multiple-Choice (MCQ) questions: type="mcq", options array, correct_answer
- {min(2, num_questions - 2)} Subjective questions: type="subjective", no options/correct_answer, just question + explanation

SUBJECTIVE QUESTIONS should:
- Ask students to explain a concept in their own words
- Require analysis, synthesis, or application
- Test deeper understanding beyond memorization
- Example: "Explain how X relates to Y and provide an example"
"""
            else:
                question_type_instruction = f"""
**ALL {num_questions} questions should be Multiple-Choice (MCQ)**:
- Each must have: type="mcq", question, options (4 strings), correct_answer (A/B/C/D)
"""
            
            # Generate quiz using LLM - grounded in what was taught and student's learning context
            segment_context = f" for segment {segment_id}" if segment_id else ""
            prompt = f"""
            You are a strict JSON generator. Create {num_questions} {difficulty} difficulty questions about {topic}{segment_context}.
{exam_style_instruction}
{student_context_section}
{question_type_instruction}
            Constraints:
            - Base ALL questions ONLY on TAUGHT CONTENT below.
            - If student context is provided, ensure questions target their doubts, unclear areas, and confusion points.
            - Return ONLY a JSON array (no prose, no code fences, no keys outside the schema).
            
            MCQ Format:
            {{
              "type": "mcq",
              "question": "...",
              "options": ["A ...", "B ...", "C ...", "D ..."],
              "correct_answer": "A",
              "explanation": "...",
              "hint": "...",
              "segment_hint": "general"
            }}
            
            SUBJECTIVE Format (if applicable):
            {{
              "type": "subjective",
              "question": "Explain/Analyze/Compare...",
              "explanation": "Model answer or key points to cover",
              "hint": "...",
              "segment_hint": "general"
            }}

            TAUGHT CONTENT (source of truth):
            {context}
            """
            
            # Call LLM and log prompt/response to session log file
            import time
            from app.utils.session_logger import get_logger
            start = time.time()
            response = await self.llm.ainvoke(prompt)
            duration = time.time() - start
            try:
                get_logger().log_llm_call(self.name, "generate_quiz", prompt, response.content, duration)
            except Exception:
                pass
            
            try:
                # Try to parse JSON, handle cases where LLM adds extra text or fences
                content = (response.content or "").strip()
                
                # CRITICAL FIX: Proper code fence removal
                if content.startswith('```'):
                    # Remove opening fence (```json or ```)
                    if content.startswith('```json'):
                        content = content[7:]  # Remove ```json
                    elif content.startswith('```'):
                        content = content[3:]  # Remove ```
                    
                    # Remove closing fence if present
                    if '```' in content:
                        content = content.split('```')[0]
                    
                    content = content.strip()
                
                # Find first JSON array if extra prose remains
                if not content.startswith('['):
                    import re
                    m = re.search(r"\[[\s\S]*\]", content)
                    if m:
                        content = m.group(0)
                
                # Validate minimal JSON
                questions = json.loads(content)
                if not isinstance(questions, list) or not questions:
                    raise ValueError("Empty or invalid quiz array")
                
                # Ensure 'question' field exists (convert question_text if needed)
                for q in questions:
                    if 'question_text' in q and 'question' not in q:
                        q['question'] = q.pop('question_text')
                    if 'segment_hint' not in q:
                        q['segment_hint'] = 'general'
            except Exception as e:
                print(f"[{self.name}] Failed to parse LLM quiz response: {e}")
                # Fallback questions
                questions = [
                    {
                        "question": f"What is a key concept in {topic}?",
                        "options": ["Option A", "Option B", "Option C", "Option D"],
                        "correct_answer": "A",
                        "explanation": f"This tests understanding of {topic}",
                        "hint": f"Think about the key concepts in {topic}"
                    }
                    for _ in range(num_questions)
                ]
            
            quiz = {
                "quiz_id": f"quiz_{concept_id}_{difficulty}",
                "topic": topic,
                "concept_id": concept_id,
                "difficulty": difficulty,
                "questions": questions,
                "why_assigned": f"Practice {difficulty} level questions for {topic}"
            }
            
            return AgentResponse(
                success=True,
                data=quiz,
                reasoning=f"Generated {len(questions)} questions for {topic}"
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Quiz generation failed: {str(e)}"
            )
    
    async def _evaluate_quiz(self, quiz: Dict, student_answers: List[str]) -> AgentResponse:
        """Evaluate student's quiz answers"""
        print(f"[{self.name}] Evaluating quiz answers")
        
        try:
            questions = quiz.get("questions", [])
            if len(student_answers) != len(questions):
                return AgentResponse(
                    success=False,
                    data={},
                    reasoning="Number of answers doesn't match number of questions"
                )
            
            # Evaluate each answer and track unclear segments
            correct_count = 0
            results = []
            unclear_segments = []  # Track which segments student struggled with
            segment_performance = {}  # Track performance by segment
            
            for i, (question, answer) in enumerate(zip(questions, student_answers)):
                q_type = question.get("type", "mcq")
                
                # Evaluate based on question type
                if q_type == "mcq":
                    is_correct = answer.upper() == question.get("correct_answer", "").upper()
                elif q_type == "subjective":
                    # For subjective, we'll need LLM evaluation later
                    # For now, mark as "needs_review"
                    is_correct = None  # Will be evaluated separately
                else:
                    is_correct = answer.upper() == question.get("correct_answer", "").upper()
                
                if is_correct:
                    correct_count += 1
                
                # Track segment performance
                segment_hint = question.get("segment_hint", "general")
                if segment_hint not in segment_performance:
                    segment_performance[segment_hint] = {"correct": 0, "total": 0}
                
                segment_performance[segment_hint]["total"] += 1
                if is_correct:
                    segment_performance[segment_hint]["correct"] += 1
                
                results.append({
                    "question_id": f"q{i+1}",
                    "student_answer": answer,
                    "correct_answer": question.get("correct_answer", "") if q_type == "mcq" else None,
                    "is_correct": is_correct,
                    "explanation": question.get("explanation", ""),
                    "hint": question.get("hint", ""),
                    "segment_hint": segment_hint,
                    "question_type": q_type
                })
            
            # Identify unclear segments (segments with <50% accuracy)
            from app.core.agent_thoughts import thoughts_tracker
            
            for segment, perf in segment_performance.items():
                if segment != "general" and perf["total"] > 0:
                    accuracy = perf["correct"] / perf["total"]
                    if accuracy < 0.5:  # Less than 50% correct
                        unclear_segments.append(segment)
                        print(f"[{self.name}] ⚠️ Unclear segment: {segment} ({perf['correct']}/{perf['total']})")
                        thoughts_tracker.add(
                            "Evaluator", 
                            f"Student struggled with '{segment}' ({int(accuracy*100)}% correct) - needs remediation",
                            "⚠️",
                            {"segment": segment, "accuracy": accuracy, "correct": perf["correct"], "total": perf["total"]}
                        )
            
            # Calculate score (only count MCQ questions for now)
            mcq_count = sum(1 for q in questions if q.get("type", "mcq") == "mcq")
            total_questions = mcq_count if mcq_count > 0 else len(questions)
            score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
            
            # Log overall performance
            if score_percentage >= 80:
                thoughts_tracker.add("Evaluator", f"Strong performance: {int(score_percentage)}% - student mastered the topic", "✅")
            elif score_percentage >= 60:
                thoughts_tracker.add("Evaluator", f"Moderate performance: {int(score_percentage)}% - some concepts need reinforcement", "📊")
            else:
                thoughts_tracker.add("Evaluator", f"Needs help: {int(score_percentage)}% - recommend re-teaching key concepts", "🔄")
            
            evaluation = {
                "quiz_id": quiz.get("quiz_id"),
                "topic": quiz.get("topic"),
                "total_questions": total_questions,
                "correct_answers": correct_count,
                "score_percentage": score_percentage,
                "results": results,
                "unclear_segments": unclear_segments,  # NEW: Segments that need remediation
                "segment_performance": segment_performance  # NEW: Detailed performance by segment
            }
            
            return AgentResponse(
                success=True,
                data=evaluation,
                reasoning=f"Evaluated quiz: {correct_count}/{total_questions} correct ({score_percentage:.1f}%)"
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Quiz evaluation failed: {str(e)}"
            )
    
    async def _update_profile(self, topic: str, score_percentage: float, concept_id: str) -> AgentResponse:
        """Update student profile with quiz results"""
        print(f"[{self.name}] Updating profile for {topic}: {score_percentage:.1f}%")
        
        try:
            # Get current proficiency
            current_proficiency = self.db.get_topic_proficiency(topic) or {}
            current_accuracy = current_proficiency.get("accuracy", 0.0)
            current_attempts = current_proficiency.get("attempts", 0)
            
            # Calculate new proficiency using exponential smoothing
            alpha = 0.3  # Learning rate
            new_accuracy = (alpha * score_percentage/100) + ((1 - alpha) * current_accuracy)
            new_attempts = current_attempts + 1
            
            # Determine strength level
            if new_accuracy >= 0.8:
                strength = "strong"
            elif new_accuracy >= 0.6:
                strength = "improving"
            else:
                strength = "weak"
            
            # Update database
            success = self.db.update_topic_proficiency(
                topic=topic,
                accuracy=new_accuracy,
                attempts=new_attempts,
                strength=strength
            )
            
            if success:
                return AgentResponse(
                    success=True,
                    data={
                        "topic": topic,
                        "new_accuracy": new_accuracy,
                        "new_attempts": new_attempts,
                        "strength": strength
                    },
                    reasoning=f"Updated {topic} proficiency: {new_accuracy:.2f} ({strength})"
                )
            else:
                return AgentResponse(
                    success=False,
                    data={},
                    reasoning="Failed to update database"
                )
                
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Profile update failed: {str(e)}"
            )

# Spaced Repetition Algorithm
class SpacedRepetitionScheduler:
    """Simple spaced repetition algorithm for practice scheduling"""
    
    def __init__(self):
        self.intervals = [1, 3, 7, 14, 30]  # Days between reviews
    
    def calculate_next_review(self, topic: str, proficiency: float, attempts: int) -> datetime:
        """Calculate next review date based on proficiency and attempts"""
        # Determine interval based on proficiency
        if proficiency >= 0.9:
            interval_days = self.intervals[-1]  # 30 days
        elif proficiency >= 0.8:
            interval_days = self.intervals[-2]  # 14 days
        elif proficiency >= 0.6:
            interval_days = self.intervals[-3]  # 7 days
        elif proficiency >= 0.4:
            interval_days = self.intervals[-4]  # 3 days
        else:
            interval_days = self.intervals[0]   # 1 day
        
        # Adjust based on number of attempts
        if attempts > 5:
            interval_days = min(interval_days * 1.5, 30)  # Cap at 30 days
        
        return datetime.now() + timedelta(days=interval_days)
    
    def get_topics_for_review(self, student_profile: Dict[str, Any]) -> List[str]:
        """Get topics that are due for review"""
        now = datetime.now()
        due_topics = []
        
        for topic, data in student_profile.items():
            if isinstance(data, dict) and "last_practiced" in data:
                last_practiced = datetime.fromisoformat(data["last_practiced"])
                next_review = self.calculate_next_review(
                    topic, 
                    data.get("accuracy", 0.0), 
                    data.get("attempts", 0)
                )
                
                if now >= next_review:
                    due_topics.append(topic)
        
        return due_topics
