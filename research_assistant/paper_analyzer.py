"""
Paper analysis module using LLMs for research paper understanding.

This module provides tools for analyzing research papers using language models,
extracting key insights, and generating structured summaries. Based on techniques
from "Scientific Document Understanding" (Beltagy et al., 2019).
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union
from enum import Enum

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of paper analysis available."""
    SUMMARY = "summary"
    METHODOLOGY = "methodology"
    CONTRIBUTIONS = "contributions"
    LIMITATIONS = "limitations"
    FUTURE_WORK = "future_work"
    TECHNICAL_DETAILS = "technical_details"
    RELATED_WORK = "related_work"


@dataclass
class PaperAnalysis:
    """Structured analysis result for a research paper."""
    paper_id: str
    paper_title: str
    analysis_type: AnalysisType
    
    # Core analysis fields
    summary: Optional[str] = None
    key_contributions: Optional[List[str]] = None
    methodology: Optional[str] = None
    technical_approach: Optional[str] = None
    experimental_results: Optional[Dict[str, Any]] = None
    limitations: Optional[List[str]] = None
    future_directions: Optional[List[str]] = None
    
    # Extracted entities
    datasets_used: Optional[List[str]] = None
    metrics_reported: Optional[Dict[str, float]] = None
    baselines_compared: Optional[List[str]] = None
    
    # Meta information
    confidence_score: Optional[float] = None
    analysis_timestamp: Optional[str] = None
    model_used: Optional[str] = None


class PaperAnalyzer:
    """
    Analyzes research papers using LLMs to extract structured information.
    
    Implements prompt engineering techniques optimized for scientific text
    understanding, including few-shot learning and chain-of-thought reasoning.
    """
    
    DEFAULT_MODEL = "gpt-4-turbo-preview"
    
    # System prompts for different analysis types
    SYSTEM_PROMPTS = {
        AnalysisType.SUMMARY: """You are an expert research paper analyst. 
        Provide concise, technical summaries that capture the essence of the paper.
        Focus on the problem, approach, and key results.""",
        
        AnalysisType.METHODOLOGY: """You are an expert in research methodology.
        Analyze the technical approach, experimental design, and implementation details.
        Be specific about algorithms, architectures, and techniques used.""",
        
        AnalysisType.CONTRIBUTIONS: """You are a research contribution assessor.
        Identify and explain the novel contributions of this work.
        Distinguish between claimed contributions and actual novel elements.""",
        
        AnalysisType.LIMITATIONS: """You are a critical research reviewer.
        Identify limitations, potential issues, and unstated assumptions.
        Consider both technical and practical limitations."""
    }
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        """
        Initialize the paper analyzer.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
            temperature: Sampling temperature (lower = more focused)
            max_tokens: Maximum tokens in response
        """
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def analyze_paper(
        self,
        paper_text: str,
        paper_id: str,
        paper_title: str,
        analysis_type: AnalysisType = AnalysisType.SUMMARY,
        additional_context: Optional[str] = None
    ) -> PaperAnalysis:
        """
        Analyze a research paper using LLM.
        
        Args:
            paper_text: Full text or abstract of the paper
            paper_id: Unique identifier for the paper
            paper_title: Title of the paper
            analysis_type: Type of analysis to perform
            additional_context: Optional additional context
            
        Returns:
            Structured PaperAnalysis object
        """
        # Construct the analysis prompt
        prompt = self._construct_prompt(
            paper_text, 
            analysis_type, 
            additional_context
        )
        
        # Get system prompt for analysis type
        system_prompt = self.SYSTEM_PROMPTS.get(
            analysis_type, 
            self.SYSTEM_PROMPTS[AnalysisType.SUMMARY]
        )
        
        try:
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Create analysis object
            analysis = PaperAnalysis(
                paper_id=paper_id,
                paper_title=paper_title,
                analysis_type=analysis_type,
                model_used=self.model
            )
            
            # Map results to analysis fields
            self._map_results_to_analysis(analysis, result, analysis_type)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze paper {paper_id}: {e}")
            raise
            
    def _construct_prompt(
        self,
        paper_text: str,
        analysis_type: AnalysisType,
        additional_context: Optional[str] = None
    ) -> str:
        """Construct analysis prompt based on type."""
        
        base_prompt = f"""Analyze the following research paper and provide a structured JSON response.

Paper content:
{paper_text}

"""
        
        if additional_context:
            base_prompt += f"\nAdditional context: {additional_context}\n"
            
        # Add specific instructions based on analysis type
        if analysis_type == AnalysisType.SUMMARY:
            base_prompt += """
Provide a comprehensive summary including:
{
    "summary": "2-3 paragraph technical summary",
    "key_contributions": ["list of main contributions"],
    "technical_approach": "brief description of the approach",
    "main_results": "key experimental results",
    "confidence_score": 0.0-1.0
}"""
            
        elif analysis_type == AnalysisType.METHODOLOGY:
            base_prompt += """
Analyze the methodology and provide:
{
    "methodology": "detailed methodology description",
    "technical_approach": "specific techniques and algorithms",
    "experimental_setup": "experiment design and setup",
    "datasets_used": ["list of datasets"],
    "baselines_compared": ["list of baseline methods"],
    "implementation_details": "key implementation details",
    "confidence_score": 0.0-1.0
}"""
            
        elif analysis_type == AnalysisType.CONTRIBUTIONS:
            base_prompt += """
Identify the contributions and provide:
{
    "key_contributions": ["detailed list of novel contributions"],
    "technical_novelty": "what makes this work novel",
    "improvements_over_prior_work": "specific improvements claimed",
    "potential_impact": "potential impact on the field",
    "confidence_score": 0.0-1.0
}"""
            
        elif analysis_type == AnalysisType.LIMITATIONS:
            base_prompt += """
Analyze limitations and provide:
{
    "limitations": ["list of identified limitations"],
    "unstated_assumptions": ["potential unstated assumptions"],
    "potential_issues": ["potential technical or practical issues"],
    "scope_constraints": "constraints on applicability",
    "future_directions": ["suggested future work"],
    "confidence_score": 0.0-1.0
}"""
            
        return base_prompt
        
    def _map_results_to_analysis(
        self,
        analysis: PaperAnalysis,
        results: Dict[str, Any],
        analysis_type: AnalysisType
    ) -> None:
        """Map LLM results to PaperAnalysis fields."""
        
        # Common fields
        analysis.confidence_score = results.get("confidence_score", 0.5)
        
        # Map based on analysis type
        if analysis_type == AnalysisType.SUMMARY:
            analysis.summary = results.get("summary")
            analysis.key_contributions = results.get("key_contributions", [])
            analysis.technical_approach = results.get("technical_approach")
            
        elif analysis_type == AnalysisType.METHODOLOGY:
            analysis.methodology = results.get("methodology")
            analysis.technical_approach = results.get("technical_approach")
            analysis.datasets_used = results.get("datasets_used", [])
            analysis.baselines_compared = results.get("baselines_compared", [])
            
        elif analysis_type == AnalysisType.CONTRIBUTIONS:
            analysis.key_contributions = results.get("key_contributions", [])
            
        elif analysis_type == AnalysisType.LIMITATIONS:
            analysis.limitations = results.get("limitations", [])
            analysis.future_directions = results.get("future_directions", [])
            
    async def batch_analyze(
        self,
        papers: List[Dict[str, str]],
        analysis_type: AnalysisType = AnalysisType.SUMMARY,
        max_concurrent: int = 5
    ) -> List[PaperAnalysis]:
        """
        Analyze multiple papers concurrently.
        
        Args:
            papers: List of dicts with 'id', 'title', and 'text' keys
            analysis_type: Type of analysis to perform
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            List of PaperAnalysis objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(paper: Dict[str, str]):
            async with semaphore:
                return await self.analyze_paper(
                    paper["text"],
                    paper["id"],
                    paper["title"],
                    analysis_type
                )
                
        tasks = [analyze_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        analyses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to analyze paper {papers[i]['id']}: {result}")
            else:
                analyses.append(result)
                
        return analyses
        
    async def extract_metrics(
        self,
        paper_text: str,
        paper_id: str
    ) -> Dict[str, Any]:
        """
        Extract specific metrics and results from paper.
        
        Args:
            paper_text: Paper text (results section preferred)
            paper_id: Paper identifier
            
        Returns:
            Dictionary of extracted metrics
        """
        prompt = """Extract all numerical results and metrics from this paper.
Return a JSON object with:
{
    "metrics": {
        "metric_name": value,
        ...
    },
    "comparisons": {
        "our_method": value,
        "baseline_name": value,
        ...
    },
    "datasets": ["list of datasets"],
    "tables": ["description of result tables"]
}

Paper text:
""" + paper_text
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting numerical results from research papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for accuracy
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Failed to extract metrics from {paper_id}: {e}")
            return {}
            
    async def generate_summary_embedding(
        self,
        analysis: PaperAnalysis
    ) -> List[float]:
        """
        Generate embedding for paper analysis for similarity search.
        
        Args:
            analysis: PaperAnalysis object
            
        Returns:
            Embedding vector
        """
        # Combine relevant fields for embedding
        text_parts = []
        
        if analysis.summary:
            text_parts.append(f"Summary: {analysis.summary}")
        if analysis.key_contributions:
            text_parts.append(f"Contributions: {', '.join(analysis.key_contributions)}")
        if analysis.technical_approach:
            text_parts.append(f"Approach: {analysis.technical_approach}")
            
        combined_text = "\n".join(text_parts)
        
        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=combined_text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []