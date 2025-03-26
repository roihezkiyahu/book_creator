import logging
import yaml
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_prompt_template(template_name: str) -> Dict[str, Any]:
    """
    Loads a prompt template from the configured templates directory.
    
    Args:
        template_name: Name of the template file without the .yaml extension.
        
    Returns:
        Dict[str, Any]: The loaded prompt template.
        
    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    prompt_template_path = Path(f"book_creator/prompts/{template_name}.yaml")
    
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
    
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        prompt_template = yaml.safe_load(f)
    
    return prompt_template

def setup_directory_structure(base_path: Path, story_name: str) -> Dict[str, str]:
    """
    Creates the directory structure for a book project.
    
    Args:
        base_path: Base directory path where the book folder will be created.
        story_name: Name of the story, used as the folder name.
        
    Returns:
        Dict[str, str]: Dictionary with normalized paths for each folder.
    """
    book_output_folder = base_path / story_name
    book_output_folder.mkdir(parents=True, exist_ok=True)
    
    text_adjacent_folder = book_output_folder / "text_adjacent"
    text_adjacent_folder.mkdir(parents=True, exist_ok=True)
    
    book_output_path = str(book_output_folder)
    text_adjacent_path = str(text_adjacent_folder)
    
    book_output_path_norm = book_output_path.replace('\\', '/')
    text_adjacent_path_norm = text_adjacent_path.replace('\\', '/')
    
    logger.info(f"Created book folder at: {book_output_folder}")
    logger.info(f"Created text_adjacent folder at: {text_adjacent_folder}")
    
    return {
        "book_path": book_output_path_norm,
        "text_adjacent_path": text_adjacent_path_norm
    }

def format_agent_tasks(story_name: str, book_path: str, text_adjacent_path: str, 
                       num_pages: int, lines_per_page: int, style_text: str) -> Dict[str, str]:
    """
    Formats task descriptions for each agent.
    
    Args:
        story_name: Name of the story.
        book_path: Path to the book folder.
        text_adjacent_path: Path to the text_adjacent folder.
        num_pages: Number of pages in the book.
        lines_per_page: Number of lines per page.
        style_text: Artistic style description for the images.
        
    Returns:
        Dict[str, str]: Dictionary with task descriptions for each agent.
    """
    story_agent_task = f"""Create a children's story about {story_name}. 
    The story should be {num_pages} pages long with {lines_per_page} lines per page. 
    Include detailed character descriptions suitable for illustrations. 
    Save the complete story to '{book_path}/story_details.txt'."""

    images_agent_task = f"""Create illustrations for '{story_name}'. 
    Use the story content from '{book_path}/story_details.txt' as the story_and_character_metadata parameter. 
    Generate a cover image and {num_pages} page illustrations with consistent character appearance. 
    Use {style_text}. 
    Save images to the images subfolder as '{book_path}/images/cover.jpg' and '{book_path}/images/page1.jpg', etc. 
    You must generate all images in order to complete the task. 
    CRITICAL WORKFLOW: 1) Read story_details.txt first, 2) Create detailed prompts, 
    3) Use validate_image_prompts EXACTLY ONCE, 4) Make ONE round of improvements based on critical feedback, 
    5) PROCEED DIRECTLY to generate_images_batch to create all images - post-processing will handle minor issues."""

    merge_agent_task = f"""Create text-adjacent layouts for '{story_name}'. 
    Use the story content from '{book_path}/story_details.txt'. 
    The images are at: '{book_path}/images/cover.jpg', '{book_path}/images/page1.jpg', etc. 
    Analyze each image for optimal text placement. 
    Place text beside images with font size 60 and white background. 
    Save layouts to '{text_adjacent_path}/cover.jpg' and '{text_adjacent_path}/page1.jpg', etc. 
    IMPORTANT VERIFICATION: After creating layouts, list the contents of the '{text_adjacent_path}' directory 
    to confirm that files were successfully created."""
    return {
        "story_agent_task": story_agent_task,
        "images_agent_task": images_agent_task,
        "merge_agent_task": merge_agent_task
    }

def prepare_enhanced_prompt(prompt: str, story_name: str, story_description: str, 
                           num_pages: int, lines_per_page: int, 
                           artistic_style: str, book_path: str, 
                           text_adjacent_path: str, agent_tasks: Dict[str, str]) -> str:
    """
    Prepares the enhanced prompt for the book creator agent.
    
    Args:
        prompt: Original prompt template.
        story_name: Name of the story.
        story_description: Description of the story.
        num_pages: Number of pages in the book.
        lines_per_page: Number of lines per page.
        artistic_style: Artistic style description.
        book_path: Path to the book folder.
        text_adjacent_path: Path to the text_adjacent folder.
        agent_tasks: Dictionary with task descriptions for each agent.
        
    Returns:
        str: Enhanced prompt for the book creator agent.
    """
    escaped_prompt = prompt.replace("{", "{{").replace("}", "}}")
    
    formattable_prompt = (escaped_prompt
                         .replace("{{num_pages}}", "{num_pages}")
                         .replace("{{lines_per_page}}", "{lines_per_page}")
                         .replace("{{story_name}}", "{story_name}"))
    
    formatted_prompt = formattable_prompt.format(
        num_pages=num_pages,
        lines_per_page=lines_per_page,
        story_name=story_name
    )
    
    enhanced_prompt = f"""
    # BOOK CREATION PROJECT: {story_name}
    
    ## PROJECT CONFIGURATION
    - Title: {story_name}
    - Story: {story_description if story_description else "Default: Child loves wild animals"}
    - Format: {num_pages} pages × {lines_per_page} lines/page
    - Style: {artistic_style if artistic_style else "Default: Watercolor, soft colors, simple shapes"}
    - Base Output Path: {book_path}
    - Layout Output Path: {text_adjacent_path}
    
    ## IMPORTANT: As the Book Creator Manager:
    - NEVER use tools directly that managed agents use - always delegate tasks to managed agents
    - DELEGATE ALL operations to the appropriate specialized agent
    - Your job is solely to coordinate the workflow, not perform operations

    ## AGENT EXAMPLES - HOW TO CALL AGENTS
    Agents must be called using the Action format with proper JSON structure.
    
    When you need to create the story:
    
    Action:
    {{
      "name": "story_agent",
      "arguments": {{
        "task": "{agent_tasks['story_agent_task']}"
      }}
    }}
    
    
    When you need to create images:
    
    Now that we have the story, I need to create illustrations.
    
    Action:
    {{
      "name": "images_agent",
      "arguments": {{
        "task": "{agent_tasks['images_agent_task']}"
      }}
    }}
    
    ## CRITICAL IMAGES WORKFLOW REMINDER
    - The images_agent should ONLY use validate_image_prompts ONCE
    - After validation, the agent should make ONE round of improvements, then go DIRECTLY to generating images
    - NO LOOPS are allowed in the validation process - "good enough" is better than perfect
    - The generate_images_batch tool applies post-processing, so minor validation issues can be ignored
    
    When you need to create layouts:
    
    Now that we have both the story and images, I need to create the layouts.
    
    Action:
    {{
      "name": "merge_agent",
      "arguments": {{
        "task": "{agent_tasks['merge_agent_task']}"
      }}
    }}
    
    
    ## PATH REFERENCE
    - Story Output: {book_path}/story_details.txt
    - Image Outputs: {book_path}/images/cover.jpg and {book_path}/images/page1.jpg, etc.
    - Layout Outputs: {text_adjacent_path}/cover.jpg and {text_adjacent_path}/page1.jpg, etc.
    
    ## CRITICAL VERIFICATION STEPS
    1. AFTER merge_agent completes, use list_directory_tree to verify that the text_adjacent folder contains files
    2. If text_adjacent folder exists but is empty, call list_directory to check its contents specifically
    3. If no files are found in text_adjacent folder, call merge_agent again
    4. NEVER report "DONE" until verified that text_adjacent folder actually contains layout files
    5. Your task is complete ONLY when all these exist:
       - story_details.txt in the main folder
       - All images in the images folder
       - Layout files in the text_adjacent folder (not just the empty folder)
    
    ## FINAL VERIFICATION REQUIRED
    Before returning "DONE", you MUST:
    1. Use list_directory_tree to check all folders
    2. Then use list_directory specifically on the text_adjacent folder to verify files exist
    3. Only if files are confirmed in the text_adjacent folder can you return "DONE"
    4. If text_adjacent folder is empty, you MUST run merge_agent again
    
    {formatted_prompt}
    """
    
    return enhanced_prompt 