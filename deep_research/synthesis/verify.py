import json
import re

from loguru import logger

from deep_research.core.text import response_text


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
            result_content = response_text(response)

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

async def add_verification_note(ctx, comprehensive_answer):
    """Add a note about strikethrough citations if any were flagged"""
    state = ctx.state.get_state(ctx.conversation_id)
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

