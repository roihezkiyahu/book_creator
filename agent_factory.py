import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from smolagents import HfApiModel, ToolCallingAgent, LiteLLMModel

from config import CONFIG
from tools import (save_text, read_text_file, read_image_file, list_directory, generate_image,
                  add_text_to_image, add_text_next_to_image, create_folder, load_image,
                  analyze_image_busy_areas, validate_image_consistency, generate_images_batch,
                  preprocess_image_prompts, list_directory_tree, validate_image_prompts)
from utils import load_prompt_template

logger = logging.getLogger(__name__)

load_dotenv()

def setup_environment() -> None:
    """
    Set up the environment variables and logging configuration.
    
    Returns:
        None
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.getLogger('smolagents.local_python_executor').setLevel(logging.ERROR)
    
    os.environ['OPENAI_API_KEY'] = os.getenv('OPEN_AI_API_KEY')

def initialize_models() -> Dict[str, Any]:
    """
    Initialize the LLM models used by the agents.
    
    Returns:
        Dict[str, Any]: Dictionary containing initialized models
    """
    model = HfApiModel()
    HF_TOKEN = os.getenv('HF_TOKEN')
    model_openai = LiteLLMModel(model_id=CONFIG["models"]["default"], api_key=os.getenv('OPEN_AI_API_KEY'))
    
    logger.info("Loaded models.")
    
    return {
        "hf_model": model,
        "openai_model": model_openai
    }

def create_story_agent(model: Any) -> ToolCallingAgent:
    """
    Create and configure the story agent.
    
    Args:
        model: The LLM model to use for the agent.
        
    Returns:
        ToolCallingAgent: Configured story agent
    """
    story_agent_prompt_templates = load_prompt_template("story_creator_prompt_tool_calling")
    
    story_agent = ToolCallingAgent(
        model=model,
        max_steps=CONFIG["max_steps"]["story_agent"],
        name="story_agent",
        prompt_templates=story_agent_prompt_templates,
        description="""An agent that is responsible for creating engaging, age-appropriate stories for children.""",
        tools=[save_text, read_text_file, list_directory, create_folder, list_directory_tree],
        planning_interval=CONFIG["planning_intervals"]["story_agent"],
        verbosity_level=CONFIG["verbosity_levels"]["story_agent"]
    )
    
    logger.info("Story agent created.")
    return story_agent

def create_images_agent(model: Any) -> ToolCallingAgent:
    """
    Create and configure the images agent.
    
    Args:
        model: The LLM model to use for the agent.
        
    Returns:
        ToolCallingAgent: Configured images agent
    """
    images_agent_prompt_templates = load_prompt_template("image_creator_prompt_tool_calling")
    
    images_agent = ToolCallingAgent(
        model=model,
        tools=[read_text_file, generate_image, generate_images_batch, 
               validate_image_prompts, validate_image_consistency, 
               create_folder, list_directory_tree],
        max_steps=CONFIG["max_steps"]["images_agent"],
        name="images_agent",
        planning_interval=CONFIG["planning_intervals"]["images_agent"],
        prompt_templates=images_agent_prompt_templates,
        description="""An agent that is dedicated to creating high-quality, consistent illustrations for children's books.""",
        verbosity_level=CONFIG["verbosity_levels"]["images_agent"]
    )
    
    logger.info("Images agent created.")
    return images_agent

def create_merge_agent(model: Any) -> ToolCallingAgent:
    """
    Create and configure the merge agent.
    
    Args:
        model: The LLM model to use for the agent.
        
    Returns:
        ToolCallingAgent: Configured merge agent
    """
    merge_agent_prompt_templates_tool_calling = load_prompt_template("merge_agent_tool_calling")
    
    merge_agent = ToolCallingAgent(
        model=model,
        tools=[save_text, read_text_file, read_image_file, list_directory, 
               add_text_next_to_image, create_folder,
               validate_image_consistency, list_directory_tree],
        max_steps=CONFIG["max_steps"]["merge_agent"],
        name="merge_agent",
        prompt_templates=merge_agent_prompt_templates_tool_calling,
        planning_interval=CONFIG["planning_intervals"]["merge_agent"],
        verbosity_level=CONFIG["verbosity_levels"]["merge_agent"],
        description="""An agent that is responsible for creating a visually appealing, readable children's book with text adjacent to images""",
    )
    
    logger.info("Merge agent created.")
    return merge_agent

def create_book_creator_agent(model: Any, managed_agents: List[ToolCallingAgent]) -> ToolCallingAgent:
    """
    Create and configure the main book creator agent.
    
    Args:
        model: The LLM model to use for the agent.
        managed_agents: List of agents managed by the book creator agent.
        
    Returns:
        ToolCallingAgent: Configured book creator agent
    """
    tool_calling_manager_prompt_templates = load_prompt_template("tool_calling_manager")
    
    book_creator_agent = ToolCallingAgent(
        model=model,
        tools=[read_text_file, list_directory, read_image_file, 
               create_folder, 
               validate_image_consistency, list_directory_tree],
        managed_agents=managed_agents,
        verbosity_level=CONFIG["verbosity_levels"]["book_creator_agent"],
        max_steps=CONFIG["max_steps"]["book_creator_agent"],
        name="Book Creator Agent",
        description="""An agent that is responsible for overseeing the entire book creation process with a focus on text-adjacent layouts""",
        prompt_templates=tool_calling_manager_prompt_templates,
        planning_interval=CONFIG["planning_intervals"]["book_creator_agent"]
    )
    
    logger.info("Book creator agent (tool calling) created.")
    return book_creator_agent

def initialize_all_agents() -> ToolCallingAgent:
    """
    Initialize and connect all agents for the book creation process.
    
    Returns:
        ToolCallingAgent: The fully configured book creator agent ready to run
    """
    setup_environment()
    models = initialize_models()
    model_openai = models["openai_model"]
    
    story_agent = create_story_agent(model_openai)
    images_agent = create_images_agent(model_openai)
    merge_agent = create_merge_agent(model_openai)
    
    book_creator_agent = create_book_creator_agent(
        model_openai,
        managed_agents=[merge_agent, story_agent, images_agent]
    )
    
    return book_creator_agent 