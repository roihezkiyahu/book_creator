"""
Configuration settings for the book_creator package.
"""

# Configuration settings
CONFIG = {
    "models": {
        "default": "gpt-4o-mini", #"o3-mini",#"gpt-4o-mini",
        "image_generation": {
            "model": "dall-e-3",
            "size": "1024x1024"
        }
    },
    "output_folder": "book_creator_output",
    "prompt_templates_path": "book_creator/prompts",
    "max_steps": {
        "story_agent": 10,
        "images_agent": 30,
        "merge_agent": 30,
        "book_creator_agent": 100
    },
    "verbosity_levels": {
        "story_agent": 2,
        "images_agent": 2,
        "merge_agent": 2,
        "book_creator_agent": 2
    },
    "planning_intervals": {
        "merge_agent": 3,
        "story_agent": 3,
        "images_agent": 3,
        "book_creator_agent": 5
    },
    "authorized_imports": {
        "merge_agent": [
            "PIL", "reportlab", "fpdf", "PyPDF2", "bs4", "os", "urllib", "IPythonImage", "numpy", 
            "io", "tempfile", "pathlib", "ImageDraw", "ImageFont", "Image"
        ],
        "book_creator_agent": [
            "plotly", "json", "pandas", "numpy", "bs4", "os", "urllib", "IPythonImage", "numpy",
            "PIL", "reportlab", "io", "tempfile", "pathlib", "ImageDraw", "ImageFont", "Image"
        ]
    }
} 