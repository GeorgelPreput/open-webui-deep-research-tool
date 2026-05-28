import logging
import math
import re
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from deep_research.core.types import RunContext
from deep_research.progress.events import MessageEvent, StatusEvent
from deep_research.semantics.eigendecomposition import create_semantic_transformation
from deep_research.semantics.embeddings import get_embedding
from deep_research.semantics.preference import translate_pdv_to_words

logger = logging.getLogger("deep_research.research")


async def _emit_status(ctx: RunContext, level: str, message: str, done: bool = False) -> None:
    await ctx.events.emit(StatusEvent(description=message, level=level, done=done))


async def _emit_message(ctx: RunContext, content: str) -> None:
    await ctx.events.emit(MessageEvent(content=content))


async def group_replacement_topics(ctx: RunContext, replacement_topics):
    """Group replacement topics semantically into groups of 2-4 topics each"""
    # Skip if too few topics
    if len(replacement_topics) <= 4:
        return [replacement_topics]  # Just one group if 4 or fewer topics

    # Get embeddings for each topic sequentially
    topic_embeddings = []
    for topic in replacement_topics:
        embedding = await get_embedding(ctx, topic)
        if embedding:
            topic_embeddings.append((topic, embedding))

    # If we don't have enough valid embeddings for grouping, use simple groups
    if len(topic_embeddings) < 3:
        logger.warning(
            "Not enough embeddings for semantic grouping, using simple groups"
        )
        # Just divide topics into groups of 4
        groups = []
        for i in range(0, len(replacement_topics), 4):
            groups.append(replacement_topics[i : i + 4])
        return groups

    try:
        # Extract embeddings into a numpy array
        embeddings_array = np.array([emb for _, emb in topic_embeddings])

        # Determine number of clusters (groups)
        total_topics = len(topic_embeddings)
        # Aim for groups of 3-4 topics each
        n_clusters = max(1, total_topics // 3)
        # Cap at a reasonable number
        n_clusters = min(n_clusters, 5)

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # pyright: ignore[reportArgumentType]
        kmeans.fit(embeddings_array)
        assert kmeans.labels_ is not None

        # Group topics by cluster
        grouped_topics = {}
        for i, (topic, _) in enumerate(topic_embeddings):
            cluster_id = kmeans.labels_[i]
            if cluster_id not in grouped_topics:
                grouped_topics[cluster_id] = []
            grouped_topics[cluster_id].append(topic)

        # Get the groups as a list
        groups_list = list(grouped_topics.values())

        # Balance any groups that are too small or large
        if len(groups_list) > 1:
            # Sort groups by size
            groups_list.sort(key=len)

            # Merge any tiny groups (fewer than 2 topics)
            while len(groups_list) > 1 and len(groups_list[0]) < 2:
                smallest = groups_list.pop(0)
                second_smallest = groups_list[0]  # Don't remove yet, just reference

                # Merge with second smallest
                groups_list[0] = second_smallest + smallest

                # Re-sort
                groups_list.sort(key=len)

            # Split any very large groups (more than 5 topics)
            for i, group in enumerate(groups_list):
                if len(group) > 5:
                    # Simple split at midpoint
                    midpoint = len(group) // 2
                    groups_list[i] = group[:midpoint]  # First half
                    groups_list.append(group[midpoint:])  # Second half

        return groups_list

    except Exception as e:
        logger.error(f"Error during topic grouping: {e}")
        # Fall back to simple grouping on error
        groups = []
        for i in range(0, len(replacement_topics), 4):
            groups.append(replacement_topics[i : i + 4])
        return groups

async def generate_group_title(ctx: RunContext, topics: list[str], user_message: str) -> str:
    """Generate a descriptive title for a group of related topics"""
    if not topics:
        return ""

    # For very small groups, just combine the topics
    if len(topics) <= 2:
        combined = " & ".join(topics)
        if len(combined) > 80:
            return combined[:77] + "..."
        return combined

    # Create a prompt to generate the group title
    title_prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant creating a concise descriptive title for a group of related research topics.
Create a short, clear title (4-8 words) that captures the common theme across these topics.
The title should be specific enough to distinguish this group from others, but general enough to encompass all topics.
DO NOT use generic phrases like "Research Group" or "Topic Group".
Respond with ONLY the title text.""",
    }

    # Create the message content with full topics
    topic_text = "\n- " + "\n- ".join(topics)

    message = {
        "role": "user",
        "content": f"""Create a concise title for this group of related research topics:
{topic_text}

These topics are part of research about: "{user_message}"

Respond with ONLY the title (4-8 words).""",
    }

    # Generate the title
    try:
        response = await ctx.client.chat_completions(
            ctx.valves.models.research_model,
            [title_prompt, message],
            temperature=0.7,
        )

        title = response["choices"][0]["message"]["content"].strip()

        # Remove quotes if present
        title = title.strip("\"'")

        # Limit length if needed
        if len(title) > 80:
            title = title[:77] + "..."

        return title
    except Exception as e:
        logger.error(f"Error generating group title: {e}")
        # Single clean fallback that uses first topic
        return f"{topics[0][:40]}... & Related Topics"

async def generate_replacement_topics(
    ctx: RunContext,
    query: str,
    kept_items: list[str],
    removed_items: list[str],
    preference_vector: dict[str, Any],
    outline_items: list[str],
) -> list[str]:
    """Generate replacement topics using semantic transformation"""
    # If nothing was removed, return empty list
    if not removed_items:
        return []

    # If nothing was kept, use the full original outline as reference
    if not kept_items:
        kept_items = outline_items

    # Calculate 80% of removed items count, rounded up
    num_replacements = math.ceil(len(removed_items) * 0.8)

    # Ensure at least one replacement
    num_replacements = max(1, num_replacements)

    logger.info(
        f"Generating {num_replacements} replacement topics (80% of {len(removed_items)} removed)"
    )

    # Create a prompt to generate replacements
    replacement_prompt = {
        "role": "system",
        "content": """You are a post-grad research assistant generating replacement topics for a research outline.
Based on the kept topics, original query, and user's preferences, generate new research topics to replace removed ones.
Each new topic should:
1. Be directly relevant to answering or addressing the original query
2. Be conceptually aligned with the kept topics
3. Avoid concepts related to removed topics and their associated themes
4. Be specific and actionable for research without devolving into hyperspecificity

Generate EXACTLY the requested number of replacement topics in a numbered list format.
Each replacement should be thoughtful and unique, exploring and expanding on different aspects of the research subject.
""",
    }

    # Extract preference information
    pdv = preference_vector.get("pdv")
    strength = preference_vector.get("strength", 0.0)
    impact = preference_vector.get("impact", 0.0)

    # Prepare the request content
    content = f"""Original query: {query}

Kept topics (conceptually preferred):
{kept_items}

Removed topics (to avoid):
{removed_items}

"""

    # Pre-compute embeddings
    if pdv is not None and impact > 0.1:
        # Get query embedding first
        query_embedding = await get_embedding(ctx, query)

        # Get kept item embeddings sequentially
        kept_embeddings = []
        for item in kept_items:
            embedding = await get_embedding(ctx, item)
            if embedding:
                kept_embeddings.append(embedding)

        # If we have enough embeddings, create a semantic transformation
        if query_embedding and len(kept_embeddings) >= 3:
            # Create a simple eigendecomposition
            try:
                # Filter out any non-array elements that could cause errors
                valid_embeddings = []
                for emb in kept_embeddings:
                    if isinstance(emb, list) or (
                        hasattr(emb, "ndim") and emb.ndim == 1
                    ):
                        valid_embeddings.append(emb)

                # Only proceed if we have enough valid embeddings
                if len(valid_embeddings) >= 3:
                    kept_array = np.array(valid_embeddings)
                    # Simple PCA
                    pca = PCA(n_components=min(3, len(valid_embeddings)))
                    pca.fit(kept_array)
                else:
                    logger.warning(
                        f"Not enough valid embeddings for PCA: {len(valid_embeddings)}/3 required"
                    )
                    return []

                assert (
                    pca.components_ is not None
                    and pca.explained_variance_ is not None
                    and pca.explained_variance_ratio_ is not None
                )
                eigen_data = {
                    "eigenvectors": pca.components_.tolist(),
                    "eigenvalues": pca.explained_variance_.tolist(),
                    "explained_variance": pca.explained_variance_ratio_.tolist(),
                }

                # Create transformation that includes PDV
                transformation = await create_semantic_transformation(ctx,
                    eigen_data, pdv=pdv
                )

                # Store for later use
                ctx.state.update_state(ctx.conversation_id, "semantic_transformations", transformation)

                logger.info(
                    "Created semantic transformation for replacement topics generation"
                )
            except Exception as e:
                logger.error(f"Error creating PCA for topic replacement: {e}")

    if pdv is not None:
        # Translate the PDV into natural language concepts
        pdv_concepts = await translate_pdv_to_words(ctx, pdv)
        if pdv_concepts:
            content += f"User preferences: The user prefers topics related to: {pdv_concepts}\n"
            if strength > 0.9:
                content += "The user has expressed a strong preference for these concepts. "
            elif strength > 0.5:
                content += "The user has expressed a moderate preference for these concepts. "
            else:
                content += "The user has expressed a slight preference for these concepts. "

    content += f"""Generate EXACTLY {num_replacements} replacement research topics in a numbered list.
These should align with the kept topics and original query, while avoiding concepts from removed topics.
Please don't include any other text in your response but the replacement topics. You don't need to justify them either.
"""

    messages = [replacement_prompt, {"role": "user", "content": content}]

    # Generate all replacements at once
    try:
        await _emit_status(ctx,
            "info", f"Generating {num_replacements} replacement topics...", False
        )

        # Generate replacements
        # Use research model for generating replacements
        research_model = ctx.valves.models.research_model
        response = await ctx.client.chat_completions(
            research_model,
            messages,
            temperature=ctx.valves.models.temperature
            * 1.1,  # Slightly higher temperature for creative replacements
        )

        if response and "choices" in response and len(response["choices"]) > 0:
            generated_text = response["choices"][0]["message"]["content"]

            # Parse the generated text to extract topics (numbered list format)
            lines = generated_text.split("\n")
            replacements = []

            for line in lines:
                # Look for numbered list items: 1. Topic description
                match = re.search(r"^\s*\d+\.\s*(.+)$", line)
                if match:
                    topic = match.group(1).strip()
                    if (
                        topic and len(topic) > 10
                    ):  # Minimum length to be a valid topic
                        replacements.append(topic)

            # Ensure we have exactly the right number of replacements
            if len(replacements) > num_replacements:
                replacements = replacements[:num_replacements]
            elif len(replacements) < num_replacements:
                # If we didn't get enough, create generic ones to fill the gap
                while len(replacements) < num_replacements:
                    missing_count = num_replacements - len(replacements)
                    await _emit_status(ctx,
                        "info",
                        f"Generating {missing_count} additional topics...",
                        False,
                    )
                    replacements.append(
                        f"Additional research on {query} aspect {len(replacements) + 1}"
                    )

            return replacements

    except Exception as e:
        logger.error(f"Error generating replacement topics: {e}")

    # Fallback - create generic replacements
    return [
        f"Alternative research topic {i + 1} for {query}"
        for i in range(num_replacements)
    ]

