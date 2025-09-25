"""
Response formatting utilities for creating structured teacher strings
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .bbox import BoundingBox, format_bbox_string
from .retrieval import TextSegment


@dataclass
class TeacherResponse:
    """Structured teacher response"""
    thought: str           # Chain-of-thought reasoning
    final_answer: str      # Final textual answer
    location: BoundingBox  # Spatial grounding
    confidence: float = 1.0


class ResponseFormatter:
    """Formatter for creating structured teacher strings"""
    
    def __init__(self, format_style: str = "cot"):
        """
        Initialize response formatter
        
        Args:
            format_style: Formatting style ("cot", "json", "structured")
        """
        self.format_style = format_style
    
    def format_teacher_string(self,
                            context: List[TextSegment],
                            answer_text: str,
                            answer_bbox: BoundingBox,
                            reasoning: Optional[str] = None) -> str:
        """
        Format complete teacher string as shown in Algorithm 1
        
        Creates: ST = FORMAT(C, Atext, BA)
        "Thought: uses retrieved context C; Final Answer: Atext; Location: BA"
        
        Args:
            context: Retrieved context segments C
            answer_text: Textual answer Atext
            answer_bbox: Answer bounding box BA
            reasoning: Optional reasoning from QA model
            
        Returns:
            Formatted teacher string ST
        """
        if self.format_style == "cot":
            return self._format_cot_style(context, answer_text, answer_bbox, reasoning)
        elif self.format_style == "json":
            return self._format_json_style(context, answer_text, answer_bbox, reasoning)
        elif self.format_style == "structured":
            return self._format_structured_style(context, answer_text, answer_bbox, reasoning)
        else:
            raise ValueError(f"Unknown format style: {self.format_style}")
    
    def _format_cot_style(self,
                         context: List[TextSegment],
                         answer_text: str,
                         answer_bbox: BoundingBox,
                         reasoning: Optional[str] = None) -> str:
        """Format in chain-of-thought style matching the paper"""
        
        # Build context summary
        if context:
            context_texts = [seg.text for seg in context]
            context_summary = "; ".join(context_texts[:3])  # Limit to first 3 for brevity
            if len(context) > 3:
                context_summary += f"; ... (+{len(context)-3} more)"
        else:
            context_summary = "No relevant context found"
        
        # Build thought process
        if reasoning:
            thought = f"uses retrieved context: {context_summary}. {reasoning}"
        else:
            thought = f"uses retrieved context: {context_summary}. Based on the retrieved text segments, I can determine the answer."
        
        # Format location
        location_str = format_bbox_string(answer_bbox, "xywh")
        
        # Create final string
        teacher_string = f"Thought: {thought}; Final Answer: {answer_text}; Location: {location_str}"
        
        return teacher_string
    
    def _format_json_style(self,
                          context: List[TextSegment],
                          answer_text: str,
                          answer_bbox: BoundingBox,
                          reasoning: Optional[str] = None) -> str:
        """Format as JSON structure"""
        
        # Build context info
        context_info = []
        for seg in context:
            context_info.append({
                "text": seg.text,
                "bbox": [seg.bbox.x, seg.bbox.y, seg.bbox.w, seg.bbox.h]
            })
        
        response_data = {
            "thought": {
                "context_summary": f"Retrieved {len(context)} relevant text segments",
                "reasoning": reasoning or "Based on the retrieved context",
                "context_details": context_info
            },
            "final_answer": answer_text,
            "location": {
                "bbox": [answer_bbox.x, answer_bbox.y, answer_bbox.w, answer_bbox.h],
                "confidence": answer_bbox.confidence
            }
        }
        
        return json.dumps(response_data, indent=2)
    
    def _format_structured_style(self,
                                context: List[TextSegment],
                                answer_text: str,
                                answer_bbox: BoundingBox,
                                reasoning: Optional[str] = None) -> str:
        """Format with clear section headers"""
        
        sections = []
        
        # Context section
        sections.append("=== CONTEXT ===")
        if context:
            for i, seg in enumerate(context[:5], 1):  # Show top 5
                sections.append(f"{i}. {seg.text}")
        else:
            sections.append("No relevant context found")
        
        # Reasoning section
        sections.append("\n=== REASONING ===")
        if reasoning:
            sections.append(reasoning)
        else:
            sections.append("Based on the retrieved context above, I can determine the answer.")
        
        # Answer section
        sections.append(f"\n=== ANSWER ===")
        sections.append(answer_text)
        
        # Location section
        sections.append(f"\n=== LOCATION ===")
        location_str = format_bbox_string(answer_bbox, "xywh")
        sections.append(f"Bounding Box: {location_str}")
        sections.append(f"Confidence: {answer_bbox.confidence:.3f}")
        
        return "\n".join(sections)
    
    def parse_teacher_string(self, teacher_string: str) -> Optional[TeacherResponse]:
        """
        Parse teacher string back to structured format
        
        Args:
            teacher_string: Formatted teacher string
            
        Returns:
            Parsed TeacherResponse or None if parsing fails
        """
        if self.format_style == "cot":
            return self._parse_cot_style(teacher_string)
        elif self.format_style == "json":
            return self._parse_json_style(teacher_string)
        elif self.format_style == "structured":
            return self._parse_structured_style(teacher_string)
        else:
            return None
    
    def _parse_cot_style(self, teacher_string: str) -> Optional[TeacherResponse]:
        """Parse chain-of-thought style string"""
        import re
        
        # Expected format: "Thought: ...; Final Answer: ...; Location: [x, y, w, h]"
        thought_match = re.search(r"Thought:\s*([^;]+)", teacher_string)
        answer_match = re.search(r"Final Answer:\s*([^;]+)", teacher_string) 
        location_match = re.search(r"Location:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", teacher_string)
        
        if not (thought_match and answer_match and location_match):
            return None
        
        thought = thought_match.group(1).strip()
        answer = answer_match.group(1).strip()
        
        # Parse bounding box
        coords = [int(x) for x in location_match.groups()]
        bbox = BoundingBox(*coords)
        
        return TeacherResponse(
            thought=thought,
            final_answer=answer,
            location=bbox
        )
    
    def _parse_json_style(self, teacher_string: str) -> Optional[TeacherResponse]:
        """Parse JSON style string"""
        try:
            data = json.loads(teacher_string)
            
            # Extract thought
            thought_data = data.get("thought", {})
            if isinstance(thought_data, str):
                thought = thought_data
            else:
                thought = thought_data.get("reasoning", "")
            
            # Extract answer and location
            answer = data.get("final_answer", "")
            location_data = data.get("location", {})
            bbox_coords = location_data.get("bbox", [0, 0, 0, 0])
            confidence = location_data.get("confidence", 1.0)
            
            bbox = BoundingBox(*bbox_coords, confidence=confidence)
            
            return TeacherResponse(
                thought=thought,
                final_answer=answer,
                location=bbox,
                confidence=confidence
            )
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _parse_structured_style(self, teacher_string: str) -> Optional[TeacherResponse]:
        """Parse structured style string"""
        sections = teacher_string.split("=== ")
        
        reasoning = ""
        answer = ""
        bbox_coords = [0, 0, 0, 0]
        confidence = 1.0
        
        for section in sections:
            if section.startswith("REASONING"):
                reasoning = section.split("===\n", 1)[-1].split("\n=== ")[0].strip()
            elif section.startswith("ANSWER"):
                answer = section.split("===\n", 1)[-1].split("\n=== ")[0].strip()
            elif section.startswith("LOCATION"):
                location_text = section.split("===\n", 1)[-1]
                # Parse bounding box coordinates
                bbox_match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", location_text)
                if bbox_match:
                    bbox_coords = [int(x) for x in bbox_match.groups()]
                # Parse confidence
                conf_match = re.search(r"Confidence:\s*([\d.]+)", location_text)
                if conf_match:
                    confidence = float(conf_match.group(1))
        
        if not answer:
            return None
        
        bbox = BoundingBox(*bbox_coords, confidence=confidence)
        
        return TeacherResponse(
            thought=reasoning,
            final_answer=answer,
            location=bbox,
            confidence=confidence
        )
    
    def format_multiple_answers(self,
                              answers_with_locations: List[Tuple[str, BoundingBox]],
                              context: List[TextSegment],
                              reasoning: Optional[str] = None) -> str:
        """
        Format multiple answers with their locations
        
        Args:
            answers_with_locations: List of (answer, bbox) pairs
            context: Retrieved context
            reasoning: Optional reasoning text
            
        Returns:
            Formatted string for multiple answers
        """
        if not answers_with_locations:
            return self.format_teacher_string(context, "No answer found", 
                                            BoundingBox(0, 0, 0, 0))
        
        if len(answers_with_locations) == 1:
            answer, bbox = answers_with_locations[0]
            return self.format_teacher_string(context, answer, bbox, reasoning)
        
        # Multiple answers
        answer_parts = []
        all_bboxes = []
        
        for i, (answer, bbox) in enumerate(answers_with_locations, 1):
            answer_parts.append(f"{i}. {answer}")
            all_bboxes.append(bbox)
        
        combined_answer = "; ".join(answer_parts)
        
        # Combine bounding boxes (encompassing box)
        if all_bboxes:
            min_x = min(bbox.x for bbox in all_bboxes)
            min_y = min(bbox.y for bbox in all_bboxes)
            max_x2 = max(bbox.x2 for bbox in all_bboxes)
            max_y2 = max(bbox.y2 for bbox in all_bboxes)
            
            combined_bbox = BoundingBox.from_xyxy(min_x, min_y, max_x2, max_y2)
        else:
            combined_bbox = BoundingBox(0, 0, 0, 0)
        
        return self.format_teacher_string(context, combined_answer, 
                                        combined_bbox, reasoning)


def format_teacher_response(context: List[TextSegment],
                           answer_text: str, 
                           answer_bbox: BoundingBox,
                           reasoning: Optional[str] = None) -> str:
    """
    Convenience function for formatting teacher response
    
    Implementation of FORMAT(C, Atext, BA) from Algorithm 1
    
    Args:
        context: Retrieved context C
        answer_text: Textual answer Atext  
        answer_bbox: Answer bounding box BA
        reasoning: Optional reasoning text
        
    Returns:
        Formatted teacher string ST
    """
    formatter = ResponseFormatter()
    return formatter.format_teacher_string(context, answer_text, 
                                         answer_bbox, reasoning)
