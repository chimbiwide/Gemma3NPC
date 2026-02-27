#!/usr/bin/env python3
"""
PIPPA Dataset Filtering Script for Gemma 3n NPC Training
Filters conversations based on OpenAI moderation scores.
"""

import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_conversation_safe(moderation_scores: Dict[str, float]) -> bool:
    """
    Determine if a conversation is safe based on moderation scores.

    Args:
        moderation_scores: Dictionary of moderation category scores

    Returns:
        bool: True if conversation is safe, False otherwise
    """
    is_safe = True

    for key, value in moderation_scores.items():
        if key == "violence" and value > 0.9:
            is_safe = False
            break
        elif key == "harassment" and value > 0.8:
            is_safe = False
            break
        elif key == "hate" and value > 0.9:
            is_safe = False
            break
        elif key == "self_harm" and value > 0.75:
            is_safe = False
            break
        elif key == "sexual_minors" and value > 0.6:
            is_safe = False
            break
        elif key == "sexual" and value > 0.8:
            is_safe = False
            break
        elif key == "self_harm_instructions" and value > 0.8:
            is_safe = False
            break

    return is_safe


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of dictionaries, one per line
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
        logger.info(f"Loaded {len(data)} entries from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise

    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(data)} entries to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise


def filter_pippa_dataset(
        scores_file: str = "pippa_score.jsonl",
        conversations_file: str = "processed_pippa.jsonl",
        output_file: str = "filter_pippa.jsonl"
) -> None:
    """
    Filter PIPPA dataset based on moderation scores.

    Args:
        scores_file: Path to moderation scores file
        conversations_file: Path to conversations file
        output_file: Path to output filtered file
    """

    logger.info("Starting PIPPA dataset filtering...")

    # Load moderation scores and conversations
    logger.info("Loading moderation scores...")
    scores = load_jsonl(scores_file)

    logger.info("Loading conversations...")
    conversations = load_jsonl(conversations_file)

    # Verify we have matching data
    if len(scores) != len(conversations):
        logger.warning(
            f"Mismatch: {len(scores)} scores vs {len(conversations)} conversations. "
            f"Will process {min(len(scores), len(conversations))} entries."
        )

    # Filter conversations
    filtered_conversations = []
    unsafe_count = 0

    min_length = min(len(scores), len(conversations))

    for i in range(min_length):
        score_data = scores[i]
        conversation_data = conversations[i]

        # Extract moderation scores from the score data
        # Adjust this if your score structure is different
        if isinstance(score_data, dict):
            moderation_scores = score_data
        else:
            logger.warning(f"Unexpected score format at index {i}: {type(score_data)}")
            continue

        # Check if conversation is safe
        if is_conversation_safe(moderation_scores):
            filtered_conversations.append(conversation_data)
        else:
            unsafe_count += 1

        # Progress logging
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{min_length} conversations...")

    # Save filtered dataset
    save_jsonl(filtered_conversations, output_file)

    # Report statistics
    original_count = min_length
    filtered_count = len(filtered_conversations)
    retention_rate = (filtered_count / original_count) * 100 if original_count > 0 else 0

    logger.info("\n" + "=" * 50)
    logger.info("FILTERING RESULTS")
    logger.info("=" * 50)
    logger.info(f"Original conversations: {original_count:,}")
    logger.info(f"Filtered conversations: {filtered_count:,}")
    logger.info(f"Removed conversations: {unsafe_count:,}")
    logger.info(f"Retention rate: {retention_rate:.1f}%")
    logger.info(f"Output saved to: {output_file}")
    logger.info("=" * 50)


def main():
    """Main function to run the filtering process."""

    # Check if input files exist
    required_files = ["pippa_score.jsonl", "processed_pippa.jsonl"]
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"Required file not found: {file_path}")
            logger.error("Please ensure both input files are in the current directory.")
            return

    try:
        filter_pippa_dataset()
        logger.info("Filtering completed successfully!")

    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        raise


if __name__ == "__main__":
    main()