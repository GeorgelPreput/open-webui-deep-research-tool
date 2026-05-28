import asyncio
import json
import re

from loguru import logger

from deep_research.config.constants import VERIFY_CITATIONS
from deep_research.progress.events import StatusEvent
from deep_research.web.fetch import fetch_content


async def verify_citation_batch(ctx, url, citations, source_content):
    """Verify a batch of citations from a single source with improved sentence context isolation"""
    try:
        # Create a verification prompt
        verify_prompt = {
            "role": "system",
            "content": """You are a post-grad research assistant verifying the accuracy of citations and cited sentences against source material.

        Examine the source content and verify accuracy of each snippet. A citation is considered verified if the source includes the cited information.

        It is imperative you actually confirm accuracy/applicability or lack of such for each citation via direct comparison to source - never try to rely on your own knowledge.

        Return your results as a JSON array with this format:
        [
          {
            "verified": true,
            "global_id": "citation_id"
          },
          {
            "verified": false,
            "global_id": "citation_id"
          }
        ]""",
        }

        # Create verification context with all citations from this source
        verify_context = (
            f"Source URL: {url}\n\nSource content excerpt:\n{source_content}...\n\n"
        )
        verify_context += "Citation contexts to verify:\n"

        for i, citation in enumerate(citations):
            text = citation.get("text", "")
            global_id = citation.get("global_id", "unknown")
            if text:
                verify_context += f'{i + 1}. "{text}" [Global ID: {global_id}]\n'

        verify_context += "\nVerify each citation context against the source content. Provide a JSON array with verification results."

        # Generate verification assessment using the research model
        response = await ctx.client.chat_completions(
            ctx.valves.models.research_model,
            [verify_prompt, {"role": "user", "content": verify_context}],
            temperature=ctx.valves.models.temperature
            * 0.2,  # 20% of normal temperature for precise verification
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            result_content = response["choices"][0]["message"]["content"]

            # Extract JSON array from the response
            try:
                # Find array pattern [...]
                array_match = re.search(r"\[(.*?)\]", result_content, re.DOTALL)
                if array_match:
                    json_array = f"[{array_match.group(1)}]"
                    verification_results = json.loads(json_array)

                    # Add additional information to each result
                    final_results = []
                    for i, result in enumerate(verification_results):
                        if i < len(citations):
                            citation = citations[i]
                            final_result = {
                                "url": url,
                                "verified": result.get("verified", False),
                                "flagged": not result.get("verified", False),
                                "citation_text": citation.get("text", ""),
                                "section": citation.get("section", ""),
                                "global_id": citation.get("global_id"),
                            }
                            final_results.append(final_result)

                    return final_results
                else:
                    # Try to parse as individual JSON objects
                    json_objects = re.findall(r"{.*?}", result_content, re.DOTALL)
                    if json_objects:
                        final_results = []
                        for i, json_str in enumerate(json_objects):
                            try:
                                result = json.loads(json_str)
                                if i < len(citations):
                                    citation = citations[i]
                                    final_result = {
                                        "url": url,
                                        "verified": result.get("verified", False),
                                        "flagged": not result.get(
                                            "verified", False
                                        ),
                                        "citation_text": citation.get("text", ""),
                                        "section": citation.get("section", ""),
                                        "global_id": citation.get("global_id"),
                                    }
                                    final_results.append(final_result)
                            except Exception:
                                continue
                        return final_results

            except Exception as e:
                logger.error(f"Error parsing verification results: {e}")

        # Fallback for failures - assume all unverified
        return [
            {
                "url": url,
                "verified": False,
                "flagged": False,
                "citation_text": citation.get("text", ""),
                "section": citation.get("section", ""),
                "global_id": citation.get("global_id"),
            }
            for citation in citations
        ]

    except Exception as e:
        logger.error(f"Error verifying batch of citations: {e}")
        return []

async def verify_citations(
    ctx, global_citation_map, citations_by_section, master_source_table
):
    """Verify citations in smaller batches"""
    if not VERIFY_CITATIONS:
        await ctx.events.emit(StatusEvent("Citation verification is disabled", "info", False))
        return {"verified": [], "flagged": []}

    # Count citations to verify
    total_citations = sum(
        len(section_citations)
        for section_citations in citations_by_section.values()
    )
    if total_citations == 0:
        await ctx.events.emit(StatusEvent("No citations to verify", "info", False))
        return {"verified": [], "flagged": []}

    # Group citations by source URL for efficient verification
    citations_by_source = {}
    for section_citations in citations_by_section.values():
        for citation in section_citations:
            url = citation.get("url")
            if not url:
                continue

            if url not in citations_by_source:
                citations_by_source[url] = []

            citations_by_source[url].append(citation)

    # Ensure verification uses global IDs by updating each citation
    for url, citations in citations_by_source.items():
        if url in global_citation_map:
            global_id = global_citation_map[url]
            for citation in citations:
                # Update marker to use global ID for verification tracking
                citation["global_id"] = global_id

    # Process numeric citations directly from section content
    state = ctx.state
    compiled_sections = state.get("section_synthesized_content", {})
    numeric_citations_by_url = {}

    # Extract all numeric citations directly from content
    for section, section_content in compiled_sections.items():
        numeric_matches = re.findall(r"\[(\d+)\]", section_content)
        for num in set(numeric_matches):
            try:
                numeric_id = int(num)
                # Find URL for this citation number in master_source_table
                for url, source_data in master_source_table.items():
                    source_id = source_data.get("id", "")
                    # Check if this source ID matches the numeric citation
                    if source_id == f"S{numeric_id}" or source_id == str(
                        numeric_id
                    ):
                        # Add to global citation map if not already there
                        if url not in global_citation_map:
                            global_citation_map[url] = len(global_citation_map) + 1

                        # Create tracking for this citation
                        if url not in numeric_citations_by_url:
                            numeric_citations_by_url[url] = []

                        # Find citation context for context checking
                        pattern = (
                            r"([^.!?]*\["
                            + re.escape(str(numeric_id))
                            + r"\][^.!?]*[.!?])"
                        )
                        context_matches = re.findall(pattern, section_content)
                        for context in context_matches:
                            numeric_citations_by_url[url].append(
                                {
                                    "marker": str(numeric_id),
                                    "raw_text": f"[{numeric_id}]",
                                    "text": context,
                                    "url": url,
                                    "section": section,
                                    "global_id": global_citation_map[url],
                                }
                            )
                        break
            except ValueError:
                continue

    # Merge numeric citations with regular ones
    for url, citations in numeric_citations_by_url.items():
        if url in citations_by_source:
            citations_by_source[url].extend(citations)
        else:
            citations_by_source[url] = citations

    # Check if we have any valid citations to verify
    if not citations_by_source:
        await ctx.events.emit(StatusEvent("No valid citations to verify", "info", False))
        return {"verified": [], "flagged": []}

    # Log beginning of verification process
        await ctx.events.emit(StatusEvent(
            f"Starting verification of {total_citations} citations from {len(citations_by_source)} sources...",
            "info",
            False,
        ))

    verification_results = {"verified": [], "flagged": []}

    # Use a semaphore to limit concurrent verifications
    semaphore = asyncio.Semaphore(1)  # Process one source at a time

    async def verify_source_with_semaphore(url, citations):
        async with semaphore:
            # Skip if URL is empty
            if not url or not citations:
                return []

            # Process citations in batches of up to 5
            all_batch_results = []
            for i in range(0, len(citations), 5):
                batch_citations = citations[i : i + 5]

                try:
                    # Get state for cache access
                    state = ctx.state
                    url_results_cache = state.get("url_results_cache", {})

                    # Check cache first
                    source_content = None
                    if url in url_results_cache:
                        source_content = url_results_cache[url]
                        logger.info(f"Using cached content for verification: {url}")

                    # If not in cache, fetch source content
                    if not source_content or len(source_content) < 200:
                        logger.info(f"Fetching content for verification: {url}")
                        source_content = await fetch_content(ctx, url)

                    if not source_content or len(source_content) < 200:
                        # If we couldn't fetch content, mark all citations as unverified
                        return [
                            {
                                "url": url,
                                "verified": False,
                                "flagged": False,
                                "citation_text": citation.get("text", ""),
                                "section": citation.get("section", ""),
                                "global_id": citation.get("global_id"),
                            }
                            for citation in batch_citations
                        ]

                    # Verify this batch of citations for this source
                    batch_results = await verify_citation_batch(ctx,
                        url, batch_citations, source_content
                    )

                    all_batch_results.extend(batch_results)

                except Exception as e:
                    logger.error(f"Error verifying source {url} batch: {e}")
                    # Mark the current batch as unverified but don't flag them
                    error_results = [
                        {
                            "url": url,
                            "verified": False,
                            "flagged": False,
                            "citation_text": citation.get("text", ""),
                            "section": citation.get("section", ""),
                            "global_id": citation.get("global_id"),
                        }
                        for citation in batch_citations
                    ]
                    all_batch_results.extend(error_results)

            return all_batch_results

    # Create verification tasks for each source
    verification_tasks = []
    for url, citations in citations_by_source.items():
        verification_tasks.append(verify_source_with_semaphore(url, citations))

    # Process all sources with semaphore control
    all_results = []

    # Execute verification tasks
    if verification_tasks:
        results = await asyncio.gather(*verification_tasks)

        # Flatten all results
        for batch_result in results:
            if batch_result:
                all_results.extend(batch_result)

    # Check for citation numbers that don't match any source
    for section, section_content in compiled_sections.items():
        numeric_matches = re.findall(r"\[(\d+)\]", section_content)
        for num in set(numeric_matches):
            try:
                numeric_id = int(num)
                # Check if this number appears in the global citation map values
                found_match = False
                for global_id in global_citation_map.values():
                    if global_id == numeric_id:
                        found_match = True
                        break

                # If no matching source found, flag this citation
                if not found_match:
                    pattern = (
                        r"([^.!?]*\["
                        + re.escape(str(numeric_id))
                        + r"\][^.!?]*[.!?])"
                    )
                    context_matches = re.findall(pattern, section_content)
                    for context in context_matches:
                        verification_results["flagged"].append(
                            {
                                "url": "",
                                "verified": False,
                                "flagged": True,
                                "citation_text": context,
                                "section": section,
                                "global_id": numeric_id,
                            }
                        )
            except ValueError:
                continue

    # Categorize results
    for result in all_results:
        if result.get("verified", False):
            verification_results["verified"].append(result)
        elif result.get("flagged", False):
            verification_results["flagged"].append(result)

    # Log completion of verification
        await ctx.events.emit(StatusEvent(
            f"Citation verification complete: {len(verification_results['verified'])} verified, {len(verification_results['flagged'])} flagged",
            "info",
            False,
        ))

    # Store verification results for later use
    ctx.state["verification_results"] = verification_results

    return verification_results

async def add_verification_note(ctx, comprehensive_answer):
    """Add a note about strikethrough citations if any were flagged"""
    state = ctx.state
    verification_results = state.get("verification_results", {})
    flagged_citations = verification_results.get("flagged", [])

    # Only add the note if we have flagged citations AND actually applied strikethrough
    citation_fixes = state.get("citation_fixes", [])
    if flagged_citations and citation_fixes:
        # Create the note
        verification_note = "\n\n## Notes on Verification\n\n"
        verification_note += "Strikethrough text indicates claims where the provided source could not be verified or was found to misrepresent the source material. The original citation number is retained for reference."
        # Check if bibliography exists in the answer
        bib_pattern = r"## Bibliography"
        bib_match = re.search(bib_pattern, comprehensive_answer)
        if bib_match:
            bib_index = bib_match.start()
            bib_content = comprehensive_answer[bib_index:]

            # Find the end of the bibliography section by looking for the next heading
            # or the research date line
            next_section_match = re.search(
                r"\n##\s+", bib_content[bib_match.end() - bib_index :]
            )
            research_date_match = re.search(
                r"\*Research conducted on:.*\*", bib_content
            )

            # Determine where to insert
            if next_section_match:
                # Insert before the next section
                insert_position = bib_index + next_section_match.start()
                comprehensive_answer = (
                    comprehensive_answer[:insert_position]
                    + verification_note
                    + comprehensive_answer[insert_position:]
                )
            elif research_date_match:
                # Insert before the research date line
                insert_position = bib_index + research_date_match.start()
                comprehensive_answer = (
                    comprehensive_answer[:insert_position]
                    + verification_note
                    + comprehensive_answer[insert_position:]
                )
            else:
                # If we can't find a good position, append to the end
                comprehensive_answer += "\n\n" + verification_note
        else:
            # If no bibliography, add at the end
            comprehensive_answer += "\n\n" + verification_note

    return comprehensive_answer

