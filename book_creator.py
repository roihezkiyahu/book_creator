import logging
from pathlib import Path
from typing import Optional

from config import CONFIG
from agent_factory import initialize_all_agents
from utils import setup_directory_structure, format_agent_tasks, prepare_enhanced_prompt

logger = logging.getLogger(__name__)

def create_book(
    prompt: str,
    title: Optional[str] = None,
    story_description: Optional[str] = None,
    num_pages: Optional[int] = 3, 
    lines_per_page: Optional[int] = 4,
    artistic_style: Optional[str] = None
) -> str:
    """Creates a children's book by coordinating specialized agents.
    
    Orchestrates the entire book creation process from story writing to
    illustration generation to layout creation, handling all necessary 
    folder creation and agent coordination.
    
    Args:
        prompt: Base prompt template for the book creation process with placeholders.
        title: Title for the story. If None, defaults to "Untitled_Story".
        story_description: Brief description of the story content. If None, uses a default.
        num_pages: Number of pages in the final book. Defaults to 3.
        lines_per_page: Number of text lines per page. Defaults to 4.
        artistic_style: Specific artistic style for illustrations. If None, uses watercolor.
        
    Returns:
        str: Result of the book creation process containing the completion status.
        
    Raises:
        Exception: If any part of the book creation process fails, with details in the error message.
    """
    logger.info("Starting book creation process...")
    logger.info(f"Prompt: {prompt[:100]}...")
    logger.info(f"Parameters: title={title}, num_pages={num_pages}, lines_per_page={lines_per_page}, " + 
                f"artistic_style={artistic_style}")
    
    try:
        story_name = title if title else "Untitled_Story"
        
        output_dir = Path(CONFIG["output_folder"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = setup_directory_structure(output_dir, story_name)
        book_output_path = paths["book_path"]
        text_adjacent_path = paths["text_adjacent_path"]
        
        style_text = artistic_style if artistic_style else "watercolor children's illustration style"
        
        agent_tasks = format_agent_tasks(
            story_name=story_name,
            book_path=book_output_path,
            text_adjacent_path=text_adjacent_path,
            num_pages=num_pages,
            lines_per_page=lines_per_page,
            style_text=style_text
        )
        
        enhanced_prompt = prepare_enhanced_prompt(
            prompt=prompt,
            story_name=story_name,
            story_description=story_description,
            num_pages=num_pages,
            lines_per_page=lines_per_page,
            artistic_style=artistic_style,
            book_path=book_output_path,
            text_adjacent_path=text_adjacent_path,
            agent_tasks=agent_tasks
        )
        
        book_creator_agent = initialize_all_agents()
        result = book_creator_agent.run(enhanced_prompt)
        
        verify_book_completion(text_adjacent_path)
        
        logger.info("Book creation completed successfully with verified text_adjacent layouts.")
        return result
    except Exception as e:
        logger.error(f"Book creation failed: {e}")
        raise

def verify_book_completion(text_adjacent_path: str) -> None:
    """
    Verify that the book creation process completed successfully by checking the text_adjacent folder.
    
    Args:
        text_adjacent_path: Path to the text_adjacent folder.
        
    Returns:
        None
        
    Raises:
        ValueError: If the text_adjacent folder is empty, indicating incomplete book creation.
    """
    text_adjacent_folder_path = Path(text_adjacent_path)
    if text_adjacent_folder_path.exists():
        text_adjacent_files = list(text_adjacent_folder_path.glob('*.jpg'))
        if not text_adjacent_files:
            logger.warning("Book creation completed but text_adjacent folder is empty! Task is not truly complete.")
            raise ValueError("INCOMPLETE: text_adjacent folder exists but contains no layout files")

if __name__ == "__main__":
    prompt = """
    # BOOK CREATION MANAGER INSTRUCTIONS
    
    You are a professional book creation manager responsible for coordinating specialized agents.
    You are allowed to call the final_answer tool only once you complete the task:
    1. There is a story_details.txt file in the output folder
    2. There are images (cover and 1 for each page) in the images output folder
    3. There are layouts (cover and 1 for each page) in the text_adjacent output folder
    
    ## CRITICAL WORKFLOW SEQUENCE
    1. FIRST: Check if story_details.txt already exists
       - If it exists, SKIP to step 2 (IMAGE CREATION) and provide the image agent with the story_details.txt content and his task
       - If it doesn't exist, proceed with step 1 (STORY CREATION)
    
    2. STORY CREATION: Use story_agent to create the story 
       - Create story ONLY ONCE
       - Save to story_details.txt
       - IMMEDIATELY proceed to step 2 after story creation

    
    3. IMAGE CREATION: Use images_agent for illustrations
       - if images exist, skip this step and move to merge agent
       - if images don't exist, proceed with this step
       - Provide full story_details.txt content to images_agent
       - Never skip this step if there are no images created yet
       - Generate only one batch of images

    4. LAYOUT CREATION: Use merge_agent for layouts
       - Only start this after images are created
       - Provide exact image paths and story content

    ## SUCCESS CRITERIA - Your task is complete ONLY when:
    - story_details.txt exists in the folder
    - All images (cover.jpg and page*.jpg) exist in the images subfolder
    - Layout images exist in the text_adjacent folder AND contain the merged text
    
    ## VERIFICATION REQUIRED
    - AFTER merge_agent completes, you MUST verify that text_adjacent folder contains files
    - Use list_directory_tree to confirm layouts exist
    - If text_adjacent folder is empty, you MUST call merge_agent again
    
    ## COMMON PITFALLS TO AVOID
    - DO NOT create additional folders after story creation
    - DO NOT call story_agent multiple times
    - DO NOT skip calling images_agent after story creation
    - NEVER attempt to create images yourself - always use images_agent
    - NEVER use any tool directly that is meant for managed agents - always delegate to the appropriate agent
    
    ## DETAILED WORKFLOW

    ### 1. STORY CREATION
    - Create a structured story with {{num_pages}} pages × {{lines_per_page}} lines per page
    - Develop consistent characters with detailed physical descriptions
    - Include educational themes and age-appropriate content
    - Save complete story to {{story_name}}/story_details.txt
    
    ### 2. IMAGE CREATION
    - Create detailed, self-contained prompts for cover + all pages
    - Include complete character details in EVERY prompt
    - Validate all prompts before generation
    - Generate all images in a single batch operation
    - Apply consistent style across all illustrations
    - Save images to {{story_name}}/images/cover.jpg and {{story_name}}/images/page1.jpg, etc. format
    - Use the images_agent for all prompt creation and image generation
    
    ### 3. LAYOUT CREATION
    - Analyze images to identify optimal text placement
    - Create text-adjacent layouts (text beside images) with a font size of 60 and background color of white
    - Ensure consistent formatting and readability
    - Maintain visual harmony between text and images
    - Save layouts to {{story_name}}/text_adjacent/cover.jpg and {{story_name}}/text_adjacent/page1.jpg, etc.
    
    ## AGENTS AND THEIR TOOL USAGE REQUIREMENTS
    - By the story agent: Use save_text to store the complete story
    - By the image agent: Use validate_image_prompts (once only) before generation, use the story_and_character_metadata for consistency
    - By the image agent: Use generate_images_batch (not individual image generation)
    - By the merge agent: Use add_text_next_to_image for layout creation
    - By all agents: Use create_folder to ensure directories exist

    ## AGENT USE REQUIREMENTS
    - Use only the story_agent for all story creation
    - Use only the images_agent for all prompt creation and image generation
    - Use only the merge_agent for all layout integration
    - DELEGATE ALL TOOL OPERATIONS TO MANAGED AGENTS - DO NOT USE TOOLS DIRECTLY
    
    ## PATH FORMAT RULES
    - ALWAYS use forward slashes (/) in all file paths, even on Windows
    - NEVER use backslashes (\\) in any file path
    - Keep paths consistent across all agent calls
    
    ## AGENT CALLING FORMAT
    To call any managed agent, use exactly this format:
    
    Action:
    {{
      "name": "agent_name",
      "arguments": {{
        "task": "Detailed instructions for the agent's task including all necessary information and file paths"
      }}
    }}
    
    """
    
    # Call the create_book function
    result = create_book(
        prompt=prompt,
        title="Geffen's Jungle Adventure",
        story_description="A story about a curious child named Geffen who goes on an adventure to meet different jungle animals in their natural habitats.",
        num_pages=3,
        lines_per_page=2,
        artistic_style="watercolor children's digital illustration style with soft colors and simple shapes"
    )
    
    print(f"Final result: {result}") 