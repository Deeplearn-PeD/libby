"""
Structured models for LLM Wiki operations.

These Pydantic models define the schema for structured LLM outputs
used during wiki ingest, query, and lint operations.
"""

from pydantic import BaseModel, Field


class KeyEntity(BaseModel):
    """An entity extracted from a source document."""

    name: str = Field(..., title="Entity name")
    entity_type: str = Field(..., title="Type of entity (person, organization, concept, etc.)")
    description: str = Field(..., title="Brief description of the entity")
    related_entities: list[str] = Field(default_factory=list, title="Related entity names")


class KeyConcept(BaseModel):
    """A concept extracted from a source document."""

    name: str = Field(..., title="Concept name")
    description: str = Field(..., title="Brief description of the concept")
    related_concepts: list[str] = Field(default_factory=list, title="Related concept names")


class SourceSummary(BaseModel):
    """Structured summary of a source document."""

    title: str = Field(..., title="Title of the source")
    summary: str = Field(..., title="Comprehensive summary of the source")
    key_takeaways: list[str] = Field(default_factory=list, title="Key takeaways")
    entities: list[KeyEntity] = Field(default_factory=list, title="Key entities mentioned")
    concepts: list[KeyConcept] = Field(default_factory=list, title="Key concepts discussed")
    contradictions: list[str] = Field(default_factory=list, title="Claims that contradict existing wiki knowledge")
    questions_raised: list[str] = Field(default_factory=list, title="Questions raised by this source")


class PageUpdate(BaseModel):
    """A single page update to apply during wiki ingest."""

    page_type: str = Field(..., title="Type of page (source, entity, concept, synthesis)")
    page_name: str = Field(..., title="Name of the page (used as filename)")
    action: str = Field(..., title="Action to take: create, update, or replace")
    reason: str = Field(..., title="Why this page needs to be created/updated")
    content_outline: str = Field(..., title="Outline of what the page should contain")


class WikiUpdatePlan(BaseModel):
    """Plan for updating the wiki after ingesting a source."""

    source_title: str = Field(..., title="Title of the source being ingested")
    pages_to_update: list[PageUpdate] = Field(
        default_factory=list, title="Pages that need to be created or updated"
    )
    pages_to_link: list[str] = Field(
        default_factory=list, title="Existing pages that should reference the new source"
    )
    synthesis_notes: str = Field(
        default="", title="Notes on how this source changes the overall synthesis"
    )


class Contradiction(BaseModel):
    """A contradiction found between wiki pages."""

    page_a: str = Field(..., title="First page")
    claim_a: str = Field(..., title="Claim on first page")
    page_b: str = Field(..., title="Second page")
    claim_b: str = Field(..., title="Claim on second page")
    severity: str = Field(..., title="Severity: minor, moderate, or major")
    resolution_hint: str = Field(..., title="Suggested way to resolve the contradiction")


class StaleClaim(BaseModel):
    """A claim that may be outdated based on newer sources."""

    page: str = Field(..., title="Page containing the stale claim")
    claim: str = Field(..., title="The potentially stale claim")
    newer_source: str = Field(..., title="Source that supersedes this claim")
    reason: str = Field(..., title="Why this claim is likely stale")


class MissingPage(BaseModel):
    """An important concept or entity that lacks a dedicated page."""

    name: str = Field(..., title="Name of the missing page")
    page_type: str = Field(..., title="Type: entity, concept, or source")
    mentioned_in: list[str] = Field(default_factory=list, title="Pages where this is mentioned")
    reason: str = Field(..., title="Why this deserves its own page")


class LintReport(BaseModel):
    """Results of a wiki lint pass."""

    orphan_pages: list[str] = Field(default_factory=list, title="Pages with no inbound links")
    contradictions: list[Contradiction] = Field(default_factory=list, title="Contradictions between pages")
    stale_claims: list[StaleClaim] = Field(default_factory=list, title="Potentially outdated claims")
    missing_pages: list[MissingPage] = Field(default_factory=list, title="Important terms lacking dedicated pages")
    broken_links: list[str] = Field(default_factory=list, title="Links pointing to non-existent pages")
    suggestions: list[str] = Field(default_factory=list, title="General suggestions for improvement")


class WikiQueryAnswer(BaseModel):
    """Structured answer to a wiki query."""

    answer: str = Field(..., title="The synthesized answer with inline citations")
    sources_used: list[str] = Field(default_factory=list, title="Wiki pages cited in the answer")
    confidence: str = Field(..., title="Confidence level: high, medium, or low")
    gaps: list[str] = Field(default_factory=list, title="Information gaps that limit the answer")
    suggested_followups: list[str] = Field(
        default_factory=list, title="Suggested follow-up questions or sources"
    )
