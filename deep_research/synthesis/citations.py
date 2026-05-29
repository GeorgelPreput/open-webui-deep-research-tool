import json
import re
from typing import Any

from loguru import logger

from deep_research.core.text import response_text
from deep_research.core.types import BibliographyData, BibliographyEntry


async def identify_and_correlate_citations(
    ctx, section_title, content, master_source_table
):
    """Identify and correlate non-numeric URL citations in a section"""
    # Create a prompt for identifying and correlating URL citations
    citation_prompt = {
        "role": "system",
        "content": """You are a master librarian identifying non-exclusively-numeric citations in research content.

        Focus ONLY on identifying non-numeric citations that appear inside brackets, such as [https://example.com] or [Reference 1].
        IGNORE all numerical citations like [1], [2], etc. as those have already been identified and correlated.

        For each non-numerical citation you identify, extract:
        1. The exact content inside the brackets
        2. The citation text exactly as it appears in the original text, including brackets
        3. The surrounding sentence to which the citation pertains
        4. A representative title for the source (10 words or less)

        Your response must only contain the identified citations as requested. Format your response as a valid JSON object with this structure:
        {
          "citations": [
            {
              "marker": "Source Name",
              "raw_text": "[Source Name]",
              "text": "surrounding sentence containing the citation",
              "url": "https://example.com",
              "suggested_title": "Descriptive Title for Source"
            },
            ...
          ]
        }""",
    }

    # Build context with full section content and source list
    citation_context = f"## Section: {section_title}\n\n"
    citation_context += content + "\n\n"

    citation_context += "## Available Sources for Citation:\n"
    for url, source_data in master_source_table.items():
        citation_context += f"{source_data['title']} ({url})\n"

    citation_context += "\nIdentify non-numeric citations, ignore numeric citations, and extract the requested structured information."

    # Generate identification and correlation
    try:
        # Use research model for citation identification with appropriate temperature
        citation_response = await ctx.client.chat_completions(
            ctx.valves.models.research_model,
            [citation_prompt, {"role": "user", "content": citation_context}],
            temperature=ctx.valves.models.temperature
            * 0.3,  # Lower temperature for precision
        )

        citation_content = response_text(citation_response)

        # Extract JSON from response
        try:
            json_str = citation_content[
                citation_content.find("{") : citation_content.rfind("}") + 1
            ]
            citation_data = json.loads(json_str)

            section_citations = []
            for citation in citation_data.get("citations", []):
                marker_text = citation.get("marker", "").strip()
                raw_text = citation.get("raw_text", "").strip()
                context = citation.get("text", "")
                matched_url = citation.get("url", "")
                suggested_title = citation.get("suggested_title", "")

                # Only add valid citations with URLs (not numerical)
                if marker_text and matched_url and not marker_text.isdigit():
                    section_citations.append(
                        {
                            "marker": marker_text,
                            "raw_text": raw_text,
                            "text": context,
                            "url": matched_url,
                            "section": section_title,
                            "suggested_title": suggested_title,
                        }
                    )

            return section_citations

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"Error parsing citation identification JSON for section {section_title}: {e}"
            )
            return []

    except Exception as e:
        logger.error(f"Error identifying citations in section {section_title}: {e}")
        return []

async def generate_bibliography(
    ctx,
    master_source_table: dict[str, dict[str, Any]],
    global_citation_map: dict[str, int],
) -> BibliographyData:
    """Generate a bibliography using sequential numbering based on actual citations in the report"""
    if not master_source_table:
        return {
            "bibliography": [],
            "title_to_global_id": {},
            "url_to_global_id": {},
        }

    # First, scan all compiled sections to find actually cited sources
    state = ctx.state.get_state(ctx.conversation_id)
    compiled_sections = state.get("section_synthesized_content", {})

    # Extract all citation numbers from the compiled text
    cited_numbers = set()
    for section_content in compiled_sections.values():
        # Find all citations in format [n] where n is a number
        citation_matches = re.findall(r"\[(\d+)\]", section_content)
        for num in citation_matches:
            cited_numbers.add(int(num))

    # Filter global_citation_map to only include cited sources
    cited_urls = {}
    for url, id_num in global_citation_map.items():
        if id_num in cited_numbers:
            cited_urls[url] = id_num

    # Sort URLs by their assigned citation ID
    sorted_urls = sorted(cited_urls.items(), key=lambda x: x[1])

    # Create bibliography entries based on cited sources only
    bibliography: list[BibliographyEntry] = []
    url_to_global_id: dict[str, int] = {}
    title_to_global_id: dict[str, int] = {}

    # Use the sequential numbers already assigned in global_citation_map
    for url, global_id in sorted_urls:
        # Get source data from master_source_table if available
        if url in master_source_table:
            source_data = master_source_table[url]
            title = source_data.get("title", "Untitled Source")
        else:
            logger.warning(
                f"URL {url} in global_citation_map not found in master_source_table"
            )
            title = f"Source {global_id}"

        # Add bibliography entry using the actual correlated URL
        bibliography.append(
            {
                "id": global_id,
                "title": title,
                "url": url,
            }
        )

        # Create mappings
        url_to_global_id[url] = global_id
        title_to_global_id[title] = global_id

    # Sort bibliography by citation ID
    bibliography.sort(key=lambda x: x["id"])

    logger.info(
        f"Generated bibliography with {len(bibliography)} cited entries (from {len(global_citation_map)} total sources)"
    )
    return {
        "bibliography": bibliography,
        "title_to_global_id": title_to_global_id,
        "url_to_global_id": url_to_global_id,
    }

async def format_bibliography_list(
    ctx, bibliography: list[BibliographyEntry]
) -> str:
    """Format the bibliography as a numbered list"""
    if not bibliography:
        return "No sources were referenced in this research."

    # Create numbered list format
    bib_list = "\n\n## Bibliography\n\n"

    # Add each bibliography entry
    for entry in bibliography:
        citation_id = entry["id"]
        title = entry["title"]
        url = entry["url"]

        # Format URL for markdown linking
        url_formatted = f"[{url}]({url})" if url.startswith("http") else url

        bib_list += f"[{citation_id}] {title}. {url_formatted}\n\n"

    return bib_list

