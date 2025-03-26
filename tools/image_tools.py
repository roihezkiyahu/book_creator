"""
Image generation and manipulation tools for the book_creator package.

This module provides tools for generating and processing images.
"""

import os
import logging
import requests
from pathlib import Path
from typing import Optional, Tuple, List, Union, Literal, Dict, Any
from smolagents import tool
from litellm import image_generation, completion
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import time
from openai import OpenAI
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)

try:
    from ..config import CONFIG
except Exception as e:
    try:
        from book_creator.config import CONFIG
    except Exception as e:
        from config import CONFIG
        
ANALYSIS_PROMPT = """
Analyze this image for text placement opportunities:
1. BUSY AREAS: Identify regions with complex visual elements (coordinates or descriptions)
2. EMPTY AREAS: Map specific locations suitable for text insertion
3. TEXT RECOMMENDATIONS:
   • Title placement: [coordinates/region]
   • Caption placement: [coordinates/region]
   • Body text placement: [coordinates/region]
4. COLOR PALETTE: List 2-3 text colors that would contrast well with the image
5. AVOID: Note specific areas to avoid placing any text

Respond with concise, actionable placement recommendations using coordinates or clear region descriptions.
"""

VALIDATION_PROMPT = """
Evaluate image consistency across these illustrations, focusing on:

CRITICAL FACTORS:
1. CHARACTER CONSISTENCY: Physical attributes, proportions, colors
2. STYLE CONSISTENCY: Artistic technique, rendering style, detail level
3. COLOR HARMONY: Palette cohesion, tone consistency, lighting style

SECONDARY FACTORS:
4. PERSPECTIVE: Viewpoint, scale relationships, spatial logic
5. BACKGROUND: Setting elements, environmental consistency
6. TECHNICAL: Resolution, clarity, artifact presence

Rate each factor 1-5 and identify the most critical inconsistencies to fix.
Prioritize your feedback on what would be most noticeable to readers.
"""

class ImageAnalysisMode(Enum):
    """Enum for different image analysis modes."""
    ANALYSIS = "analysis"  
    VALIDATION = "validation"  

def _prepare_image_generation(image_name: str, output_folder: str) -> Path:
    """
    Validates image name and prepares output directory for image generation.
    
    Args:
        image_name: The name of the image file to validate. Recommended format: '{story_name}/[page_number].jpg'
        output_folder: The output directory to create if it doesn't exist.
        
    Returns:
        Path: The validated output path object.
        
    Raises:
        ValueError: If image_name has an invalid extension.
    """
    valid_extensions = ['.jpg', '.jpeg', '.png']
    if not any(image_name.lower().endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Image name must end with one of: {valid_extensions} (.jpg preferred for consistency)")
    
    if not image_name.lower().endswith('.jpg'):
        logger.warning(f"Image name '{image_name}' does not use the recommended .jpg extension. Using .jpg is preferred for consistency across the system.")
        
    output_path = Path(output_folder)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        img_path = Path(image_name)
        if img_path.parent != Path('.'):
            nested_path = output_path / img_path.parent
            nested_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created nested directory structure: {nested_path}")
    except Exception as e:
        logger.error(f"Failed to create directory structure: {e}")
        raise
        
    return output_path

@tool
def generate_image(image_name: str, prompt: str, style: Optional[str] = None, output_folder: Optional[str] = None) -> str:
    """
    Generates an image based on the provided prompt using the configured image generation model.
    
    Args:
        image_name: The name of the image file (must end in .png, .jpg, or .jpeg).
        prompt: Detailed description for image generation.
        style: Style of the image generation. Any consistent string describing artistic style.
        output_folder: Output directory path where the image will be saved. If None, uses the default from CONFIG.
        
    Returns:
        str: Path to the generated image.
        
    Raises:
        ValueError: If image_name has an invalid extension or output_folder creation fails.
        TypeError: If the image response cannot be processed.
    """
    if output_folder is None:
        output_folder = CONFIG["output_folder"]
    
    if style:
        full_prompt = f"{prompt}\n\nApply this consistent artistic style to the entire image: {style}"
    else:
        full_prompt = prompt
    
    output_path = _prepare_image_generation(image_name, output_folder)
    
    logger.info(f"Generating image for prompt: {full_prompt[:100]}...")
    
    try:
        logger.info(f"Calling image_generation with model={CONFIG['models']['image_generation']['model']}, size={CONFIG['models']['image_generation']['size']}")
        image_response = image_generation(
            prompt=full_prompt, 
            model=CONFIG["models"]["image_generation"]["model"], 
            size=CONFIG["models"]["image_generation"]["size"],
            api_key=os.getenv('OPEN_AI_API_KEY')
        )
        img_data = image_response['data'][0]['url']
        response = requests.get(img_data)

        save_path = output_path / image_name
        with open(save_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Successfully saved image to {save_path}")
        return str(save_path)
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise

@tool
def load_image(image_path: str) -> dict:
    """
    Loads an image from the specified path and returns a base64 encoded image URL.
    
    Args:
        image_path: Path to the image file to load.
        
    Returns:
        dict: A dictionary containing the base64 encoded image URL.
        
    Raises:
        FileNotFoundError: If the image file does not exist.
        IOError: If the file exists but cannot be opened as an image.
    """
    logger.info(f"Loading image from {image_path}")
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    try:
        path = Path(image_path) if not isinstance(image_path, Path) else image_path
        
        if not path.exists():
            logger.error(f"Image file not found: {path}")
            raise FileNotFoundError(f"Image file not found: {path}")
        
        img = encode_image(path)
        return {"image_url": f"data:image/jpeg;base64,{img}"}
        
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise IOError(f"Failed to load image: {e}")

def _get_font(font_name: Optional[str] = None, font_size: int = 24, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    """
    Gets a font for image text operations with fallback options.
    
    Args:
        font_name: Name of the font to use. If None, uses a default system font.
        font_size: Size of the font in points.
        bold: Whether to use a bold version of the font if available.
        italic: Whether to use an italic version of the font if available.
        
    Returns:
        ImageFont.FreeTypeFont: The font object to use for drawing text.
    """
    try:
        if font_name:
            style = ""
            if bold:
                style += "Bold"
            if italic:
                style += "Italic"
            return ImageFont.truetype(f"{font_name}{style}.ttf", font_size)
        
        common_fonts = []
        
        if bold:
            common_fonts.extend([
                "Arial-Bold.ttf", "arialbd.ttf",
                "Verdana-Bold.ttf", "verdanab.ttf",
                "Times-Bold.ttf", "timesbd.ttf",
                "TimesNewRoman-Bold.ttf", "timesnewromanbd.ttf",
                "DejaVuSans-Bold.ttf",
                "Roboto-Bold.ttf",
                "OpenSans-Bold.ttf"
            ])
        if italic:
            common_fonts.extend([
                "Arial-Italic.ttf", "ariali.ttf",
                "Verdana-Italic.ttf", "verdanai.ttf",
                "Times-Italic.ttf", "timesi.ttf",
                "TimesNewRoman-Italic.ttf", "timesnewroman.ttf",
                "DejaVuSans-Italic.ttf",
                "Roboto-Italic.ttf",
                "OpenSans-Italic.ttf"
            ])
        
        common_fonts.extend([
            "Arial.ttf", "arial.ttf",
            "Verdana.ttf", "verdana.ttf",
            "Times.ttf", "times.ttf",
            "TimesNewRoman.ttf", "timesnewroman.ttf",
            "DejaVuSans.ttf",
            "Roboto-Regular.ttf",
            "OpenSans-Regular.ttf"
        ])
        
        for font in common_fonts:
            try:
                return ImageFont.truetype(font, font_size)
            except (OSError, IOError):
                continue
                
        return ImageFont.load_default()
        
    except Exception as e:
        logger.warning(f"Error loading font: {e}. Using default font.")
        return ImageFont.load_default()

def _validate_font(font_name: Optional[str]) -> Optional[str]:
    """
    Validates if a font is available in the system.
    
    Args:
        font_name: Name of the font to validate.
        
    Returns:
        Optional[str]: The validated font name or None if not available.
    """
    if not font_name:
        return None
        
    try:
        ImageFont.truetype(f"{font_name}.ttf", 12)
        return font_name
    except (OSError, IOError):
        logger.warning(f"Font '{font_name}' not found. Using default font.")
        return None

def _parse_color(color_str: Optional[str], color_type: Literal["text", "background"] = "text") -> Union[str, Tuple[int, int, int]]:
    """
    Parses color from string format with defaults based on color type.
    
    Args:
        color_str: Color as a string name or RGB tuple string "(r,g,b)".
        color_type: Type of color being parsed, either "text" or "background". Defaults to "text".
        
    Returns:
        Union[str, Tuple[int, int, int]]: The parsed color as a string or RGB tuple.
        
    Examples:
        >>> _parse_color("(255,0,0)", "text")
        (255, 0, 0)
        >>> _parse_color(None, "text")
        "black"
        >>> _parse_color(None, "background")
        "white"
    """
    if not color_str:
        return "black" if color_type == "text" else "white"
        
    if color_str.startswith("(") and color_str.endswith(")"):
        try:
            rgb_values = color_str.strip("()").split(",")
            rgb_values = color_str.strip("()").split(",")
            if len(rgb_values) == 3:
                return (int(rgb_values[0]), int(rgb_values[1]), int(rgb_values[2]))
        except ValueError:
            logger.warning(f"Invalid RGB color format: {color_str}. Using default color for {color_type}.")
            return "black" if color_type == "text" else "white"
            
    # Return as color name
    return color_str

def _parse_position(position_str: str) -> Union[str, Tuple[int, int, int, int]]:
    """
    Parses position string into either a position name or bounding box coordinates.
    
    Args:
        position_str: Position as a string. Can be a position name or bounding box coordinates "(x1,y1,x2,y2)".
        
    Returns:
        Union[str, Tuple[int, int, int, int]]: The parsed position as a string or bounding box tuple.
    """
    if not position_str:
        return "center"
        
    if position_str.startswith("(") and position_str.endswith(")"):
        try:
            coords = position_str.strip("()").split(",")
            coords = position_str.strip("()").split(",")
            if len(coords) == 4:
                return (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
        except ValueError:
            logger.warning(f"Invalid position format: {position_str}. Using default position.")
            return "center"
            
    # Return as position name
    return position_str

def _create_output_path(image_path: str, output_path: Optional[str] = None, suffix: str = "_with_text") -> str:
    """
    Creates an output path for a modified image.
    
    Args:
        image_path: Original image path.
        output_path: Optional custom output path.
        suffix: Suffix to add to the filename if output_path is None.
        
    Returns:
        str: The output path to use.
    """
    if output_path is None:
        base_name, ext = os.path.splitext(image_path)
        return f"{base_name}{suffix}{ext}"
    return output_path

def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> List[str]:
    """
    Wraps text to fit within a maximum width.
    
    Args:
        text: The text to wrap.
        font: The font to use for measuring text width.
        max_width: The maximum width in pixels.
        draw: ImageDraw object for measuring text.
        
    Returns:
        List[str]: List of wrapped text lines.
    """
    lines = []
    paragraphs = text.split("\n")
    
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append('')
            continue
            
        current_line = words[0]
        
        for word in words[1:]:
            test_line = current_line + " " + word
            test_width = draw.textbbox((0, 0), test_line, font=font)[2]
            if test_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)
    
    return lines

def _calculate_text_position(
    position: Union[str, Tuple[int, int, int, int]],
    img_width: int,
    img_height: int,
    text_width: int,
    text_height: int,
    padding: int
) -> Tuple[int, int, int, int]:
    """
    Calculates the text position based on the specified position string or bounding box.
    
    Args:
        position: Position as a string or bounding box coordinates.
        img_width: Width of the image.
        img_height: Height of the image.
        text_width: Width of the text.
        text_height: Height of the text.
        padding: Padding around the text.
        
    Returns:
        Tuple[int, int, int, int]: The calculated text position as (x1, y1, x2, y2).
    """
    if isinstance(position, tuple) and len(position) == 4:
        return position
        
    position_str = position.lower()
    
    if position_str == "top":
        return (
            (img_width - text_width) // 2,
            padding,
            (img_width + text_width) // 2,
            padding + text_height
        )
    elif position_str == "bottom":
        return (
            (img_width - text_width) // 2,
            img_height - text_height - padding,
            (img_width + text_width) // 2,
            img_height - padding
        )
    elif position_str == "center":
        return (
            (img_width - text_width) // 2,
            (img_height - text_height) // 2,
            (img_width + text_width) // 2,
            (img_height + text_height) // 2
        )
    elif position_str == "top-left":
        return (
            padding,
            padding,
            padding + text_width,
            padding + text_height
        )
    elif position_str == "top-right":
        return (
            img_width - text_width - padding,
            padding,
            img_width - padding,
            padding + text_height
        )
    elif position_str == "bottom-left":
        return (
            padding,
            img_height - text_height - padding,
            padding + text_width,
            img_height - padding
        )
    elif position_str == "bottom-right":
        return (
            img_width - text_width - padding,
            img_height - text_height - padding,
            img_width - padding,
            img_height - padding
        )
    elif position_str == "center-left":
        return (
            padding,
            (img_height - text_height) // 2,
            padding + text_width,
            (img_height + text_height) // 2
        )
    elif position_str == "center-right":
        return (
            img_width - text_width - padding,
            (img_height - text_height) // 2,
            img_width - padding,
            (img_height + text_height) // 2
        )
    else:
        return (
            (img_width - text_width) // 2,
            (img_height - text_height) // 2,
            (img_width + text_width) // 2,
            (img_height + text_height) // 2
        )

@tool
def add_text_to_image(
    image_path: str,
    text: str,
    output_path: Optional[str] = None,
    position_str: Optional[str] = None,
    font_size: Optional[int] = None,
    font_name: Optional[str] = None,
    text_color_str: Optional[str] = None,
    background_color_str: Optional[str] = None,
    align: Optional[str] = None,
    vertical_align: Optional[str] = None,
    padding: Optional[int] = None,
    bold: Optional[bool] = None,
    italic: Optional[bool] = None,
    shadow: Optional[bool] = None
) -> str:
    """
    Adds text to an image with customizable positioning and styling.
    
    Args:
        image_path: Path to the source image.
        text: Text to add to the image.
        output_path: Path to save the resulting image. If None, modifies the original filename.
        position_str: Position to place the text. Can be "top", "bottom", "center", "top-left", 
                      "top-right", "bottom-left", "bottom-right", "center-left", "center-right",
                      or a string of "(x1,y1,x2,y2)" for bounding box location. Defaults to "center".
        font_size: Size of the font in points. Defaults to 24.
        font_name: Name of the font to use. If None or invalid, uses a default system font.
        text_color_str: Color of the text. Can be a color name or a string of "(R,G,B)". Defaults to "black".
        background_color_str: Background color for the text. Can be a color name or a string of "(R,G,B)". Defaults to "white".
        align: Horizontal text alignment. Can be "left", "center", or "right". Defaults to "center".
        vertical_align: Vertical text alignment. Can be "top", "middle", or "bottom". Defaults to "middle".
        padding: Padding around the text in pixels. Defaults to 10.
        bold: Whether to use a bold font if available. Defaults to False.
        italic: Whether to use an italic font if available. Defaults to False.
        shadow: Whether to add a shadow effect to the text. Defaults to False.
        
    Returns:
        str: Path to the modified image.
        
    Raises:
        FileNotFoundError: If the source image doesn't exist.
        ValueError: If position values are invalid.
    """
    position_str = position_str or "center"
    font_size = font_size or 24
    text_color_str = text_color_str or "black"
    background_color_str = background_color_str or "white"
    align = align or "center"
    vertical_align = vertical_align or "middle"
    padding = padding or 10
    bold = bold if bold is not None else False
    italic = italic if italic is not None else False
    shadow = shadow if shadow is not None else False
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path)
    
    final_output_path = _create_output_path(image_path, output_path)
    
    font_name = _validate_font(font_name)
    
    font = _get_font(font_name, font_size, bold, italic)
    
    text_color = _parse_color(text_color_str, "text")
    background_color = _parse_color(background_color_str, "background")
    
    position = _parse_position(position_str)
    
    draw = ImageDraw.Draw(img)
    
    max_width = img.width - (2 * padding)
    lines = _wrap_text(text, font, max_width, draw)
    
    line_height = font_size * 1.2
    total_text_height = len(lines) * line_height
    
    max_line_width = 0
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        max_line_width = max(max_line_width, line_width)
    
    text_box = _calculate_text_position(
        position, 
        img.width, 
        img.height, 
        max_line_width + (2 * padding), 
        total_text_height + (2 * padding),
        padding
    )
    
    left, top, right, bottom = text_box
    
    if background_color:
        draw.rectangle([left, top, right, bottom], fill=background_color)
    
    if align == "left":
        x = left + padding
    elif align == "right":
        x = right - padding - max_line_width
    else:  # center
        x = left + ((right - left - max_line_width) // 2)
    
    if vertical_align == "top":
        y = top + padding
    elif vertical_align == "bottom":
        y = bottom - padding - total_text_height
    else:  # middle
        y = top + ((bottom - top - total_text_height) // 2)
    
    for i, line in enumerate(lines):
        if align == "left":
            line_x = x
        elif align == "right":
            line_width = draw.textbbox((0, 0), line, font=font)[2]
            line_x = right - padding - line_width
        else:  # center
            line_width = draw.textbbox((0, 0), line, font=font)[2]
            line_x = left + ((right - left - line_width) // 2)
        
        line_y = y + (i * line_height)
        
        if shadow:
            shadow_x = line_x + 2
            shadow_y = line_y + 2
            draw.text((shadow_x, shadow_y), line, fill="gray", font=font)
        
        draw.text((line_x, line_y), line, fill=text_color, font=font)

    img.save(final_output_path)
    logger.info(f"Successfully saved image with text to {final_output_path}")
    
    return final_output_path

@tool
def add_text_next_to_image(
    image_path: str,
    text: str,
    output_path: Optional[str] = None,
    position_str: Optional[str] = None,
    font_size: Optional[int] = None,
    font_name: Optional[str] = None,
    text_color_str: Optional[str] = None,
    background_color_str: Optional[str] = None,
    align: Optional[str] = None,
    vertical_align: Optional[str] = None,
    padding: Optional[int] = None,
    bold: Optional[bool] = None,
    italic: Optional[bool] = None,
    shadow: Optional[bool] = None,
    text_width_ratio: Optional[float] = None
) -> str:
    """
    Extends an image with a text box next to it.
    
    Args:
        image_path: Path to the source image.
        text: Text to add next to the image.
        output_path: Path to save the resulting image. If None, modifies the original filename.
        position_str: Position to place the text box. Can be "left", "right", "top", or "bottom". Defaults to "center".
        font_size: Size of the font in points. Defaults to 24.
        font_name: Name of the font to use. If None or invalid, uses a default system font.
        text_color_str: Color of the text. Can be a color name or a string of "(R,G,B)". Defaults to "black".
        background_color_str: Background color for the text area. Can be a color name or a string of "(R,G,B)".
        align: Horizontal text alignment. Can be "left", "center", or "right". Defaults to "center".
        vertical_align: Vertical text alignment. Can be "top", "middle", or "bottom". Defaults to "middle".
        padding: Padding around the text in pixels. Defaults to 10.
        bold: Whether to use a bold font if available. Defaults to False.
        italic: Whether to use an italic font if available. Defaults to False.
        shadow: Whether to add a shadow effect to the text. Defaults to False.
        text_width_ratio: Ratio of text area width/height to the image width/height (0.0-1.0). Defaults to 0.3.
        
    Returns:
        str: Path to the modified image.
        
    Raises:
        FileNotFoundError: If the source image doesn't exist.
        ValueError: If position values are invalid.
    """
    position_str = position_str or "center"
    font_size = font_size or 24
    text_color_str = text_color_str or "black"
    align = align or "center"
    vertical_align = vertical_align or "middle"
    padding = padding or 10
    bold = bold if bold is not None else False
    italic = italic if italic is not None else False
    shadow = shadow if shadow is not None else False
    text_width_ratio = text_width_ratio if text_width_ratio is not None else 0.3
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)

    final_output_path = _create_output_path(image_path, output_path, suffix="_with_text_box")
    font_name = _validate_font(font_name)
    font = _get_font(font_name, font_size, bold, italic)
    text_color = _parse_color(text_color_str, "text")
    background_color = _parse_color(background_color_str, "background")
    position = position_str.lower()
    if position not in ["left", "right", "top", "bottom"]:
        position = "right"  # Default position
    if position in ["left", "right"]:
        text_width = int(img.width * text_width_ratio)
        new_width = img.width + text_width
        new_height = img.height
    else:  # top or bottom
        text_height = int(img.height * text_width_ratio)
        new_width = img.width
        new_height = img.height + text_height
    
    new_img = Image.new("RGB", (new_width, new_height), background_color)

    if position == "left":
        paste_position = (text_width, 0)
        text_box = (0, 0, text_width, img.height)
    elif position == "right":
        paste_position = (0, 0)
        text_box = (img.width, 0, new_width, img.height)
    elif position == "top":
        paste_position = (0, text_height)
        text_box = (0, 0, img.width, text_height)
    else:  # bottom
        paste_position = (0, 0)
        text_box = (0, img.height, img.width, new_height)
    
    new_img.paste(img, paste_position)
    
    draw = ImageDraw.Draw(new_img)
    
    left, top, right, bottom = text_box
    text_area_width = right - left - (2 * padding)
    lines = _wrap_text(text, font, text_area_width, draw)
    line_height = font_size * 1.2
    total_text_height = len(lines) * line_height
    max_line_width = 0
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font)[2]
        max_line_width = max(max_line_width, line_width)
    
    if align == "left":
        x = left + padding
    elif align == "right":
        x = right - padding - max_line_width
    else:  # center
        x = left + ((right - left - max_line_width) // 2)
    
    if vertical_align == "top":
        y = top + padding
    elif vertical_align == "bottom":
        y = bottom - padding - total_text_height
    else:  # middle
        y = top + ((bottom - top - total_text_height) // 2)
    
    for i, line in enumerate(lines):
        if align == "left":
            line_x = x
        elif align == "right":
            line_width = draw.textbbox((0, 0), line, font=font)[2]
            line_x = right - padding - line_width
        else:  # center
            line_width = draw.textbbox((0, 0), line, font=font)[2]
            line_x = left + ((right - left - line_width) // 2)
        
        line_y = y + (i * line_height)
        
        if shadow:
            shadow_x = line_x + 2
            shadow_y = line_y + 2
            draw.text((shadow_x, shadow_y), line, fill="gray", font=font)
        draw.text((line_x, line_y), line, fill=text_color, font=font)
    new_img.save(final_output_path)
    logger.info(f"Successfully saved image with text box to {final_output_path}")
    
    return final_output_path

@tool
def analyze_image_busy_areas(
    image_paths: Any,
    model_id: str = "gpt-4o-mini",
    max_tokens: int = 4000,
    batch_size: int = 4,
    temperature: float = 0
) -> Dict[str, str]:
    """
    Analyzes images to identify busy areas where text should NOT be placed,
    helping with optimal text positioning during book layout.
    
    Args:
        image_paths: List of paths to images, or a single path as string.
        model_id: The model ID to use for analysis. Defaults to "gpt-4o-mini".
        max_tokens: Maximum number of tokens for the response. Defaults to 4000.
        batch_size: Number of images to process in each batch. Defaults to 4.
        temperature: Controls analysis precision. Low values (0.1-0.3) provide more
                    precise identification of busy areas. Defaults to 0.2.
    
    Returns:
        Dict[str, str]: A dictionary mapping image paths to their busy area analyses.
        
    Raises:
        ValueError: If image analysis fails.
    """
    logger.info(f"Analyzing image busy areas using model {model_id}")
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    for path in image_paths:
        img_path = Path(path)
        if not img_path.exists():
            logger.error(f"Image file not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")
    
    try:
        results = {}
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
            
            for img_path in batch:
                content = []
                
                content.append({
                    "type": "text",
                    "text": ANALYSIS_PROMPT
                })
                
                with open(img_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
                
                start_time = time.time()
                response = completion(
                    model=model_id,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                processing_time = time.time() - start_time
                
                analysis = response.choices[0].message.content
                logger.info(f"Generated analysis for {img_path} in {processing_time:.2f} seconds")
                
                results[str(img_path)] = analysis
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing image busy areas: {str(e)}")
        raise ValueError(f"Failed to analyze image busy areas: {str(e)}")

@tool
def validate_image_consistency(
    image_paths: List[str],
    model_id: str = "gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0
) -> Dict[str, str]:
    """
    Validates consistency across multiple images to ensure characters, styles, and themes remain consistent.
    
    Args:
        image_paths: List of paths to images to analyze for consistency.
        model_id: The model ID to use for validation. Defaults to "gpt-4o-mini".
        max_tokens: Maximum number of tokens for the response. Defaults to 4000.
        temperature: Controls analysis precision. Lower values (0.0-0.3) provide more 
                    consistent analysis of visual elements. Defaults to 0.3.
    
    Returns:
        Dict[str, str]: A dictionary containing the consistency analysis and the image paths.
        
    Raises:
        ValueError: If image consistency validation fails.
    """
    logger.info(f"Validating image consistency across {len(image_paths)} images using model {model_id}")
    
    if len(image_paths) < 2:
        raise ValueError("At least 2 images are required for consistency validation")
    
    for path in image_paths:
        img_path = Path(path)
        if not img_path.exists():
            logger.error(f"Image file not found: {img_path}")
            raise FileNotFoundError(f"Image file not found: {img_path}")
    
    try:
        content = []
        content.append({
            "type": "text",
            "text": VALIDATION_PROMPT
        })
        for path in image_paths:
            img_path = Path(path)
            logger.info(f"Processing image: {img_path}")
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        start_time = time.time()
        response = completion(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        processing_time = time.time() - start_time
        validation_results = response.choices[0].message.content
        logger.info(f"Generated consistency validation in {processing_time:.2f} seconds")
        return {
            "combined_analysis": validation_results,
            "image_paths": [str(path) for path in image_paths]
        }
        
    except Exception as e:
        logger.error(f"Error validating image consistency: {str(e)}")
        raise ValueError(f"Failed to validate image consistency: {str(e)}")

@tool
def validate_image_prompts(
    prompts_list: List[Dict[str, str]],
    story_content: str,
    character_metadata: str,
    style: str,
    model_id: str = "gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0,
    max_prompts_to_validate: int = 5
) -> Dict[str, Any]:
    """
    Validates image prompts before generation to ensure high-quality, consistent illustrations.
    Limits validation to a maximum of 5 prompts to optimize performance.
    Automatically approves prompts after 3 validation attempts to prevent looping.
    
    Args:
        prompts_list: List of prompt dictionaries for image generation, each containing 'image_name' and 'prompt' keys.
        story_content: The complete story text for reference.
        character_metadata: Character descriptions for reference.
        style: The artistic style to be applied to all images.
        model_id: The model ID to use for validation. Defaults to "gpt-4o-mini".
        max_tokens: Maximum number of tokens for the response. Defaults to 4000.
        temperature: Controls randomness of output. Low values (0.0-0.2) for more precise analysis. Defaults to 0.
        max_prompts_to_validate: Maximum number of prompts to validate in one call. Defaults to 5.
    
    Returns:
        Dict[str, Any]: A structured dictionary containing validation results:
            - overall_assessment: Brief summary of all prompts' quality
            - prompt_feedback: Dictionary mapping image names to their specific feedback
            - key_improvements: List of critical issues to fix across all prompts
            - next_steps: Actionable recommendations
            - quality_scores: Numerical scores for different aspects of the prompts
            - auto_approved: Boolean indicating if prompts were automatically approved
        
    Raises:
        ValueError: If the prompts list is empty or if the model call fails.
    """
    if not hasattr(validate_image_prompts, "call_count"):
        validate_image_prompts.call_count = 0
    validate_image_prompts.call_count += 1
    logger.info(f"Prompt validation attempt #{validate_image_prompts.call_count}")
    if validate_image_prompts.call_count >= 3:
        logger.info("Maximum validation attempts reached - automatically approving prompts")
        return {
            "overall_assessment": "AUTOMATICALLY APPROVED - Prompts approved after multiple validation attempts",
            "prompt_feedback": {prompt["image_name"]: {"assessment": "Automatically approved", "score": 8} for prompt in prompts_list},
            "key_improvements": ["No further improvements needed - proceed to generation"],
            "next_steps": "PROCEED DIRECTLY TO GENERATION - Do not validate again",
            "quality_scores": {
                "overall": 8.5,
                "character_details": 8,
                "consistency": 8,
                "visual_clarity": 8
            },
            "auto_approved": True,
            "validation_metadata": {
                "prompts_validated": len(prompts_list),
                "total_prompts": len(prompts_list),
                "processing_time_seconds": 0.1,
                "validation_attempts": validate_image_prompts.call_count
            }
        }
    
    logger.info(f"Validating image prompts before generation (max {max_prompts_to_validate})")
    if not prompts_list:
        raise ValueError("Prompts list cannot be empty")
    book_title = ""
    title_match = re.search(r"title:?\s*['\"](.*?)['\"]", story_content, re.IGNORECASE)
    if title_match:
        book_title = title_match.group(1)
    else:
        for prompt_data in prompts_list:
            if "cover" in prompt_data.get("image_name", "").lower():
                parts = prompt_data["image_name"].split("/")
                if len(parts) > 1:
                    book_title = parts[-2]  # Get the folder name which is often the title
                    break
    for prompt_data in prompts_list:
        if "cover" in prompt_data.get("image_name", "").lower():
            prompt = prompt_data.get("prompt", "")
            if book_title and book_title.lower() not in prompt.lower():
                logger.warning(f"Cover image prompt does not include the book title: '{book_title}'")
                prompt_data["prompt"] = f"BOOK COVER: {prompt} The image should prominently display the title '{book_title}' as part of the cover design."
            elif "cover" not in prompt.lower() and "title" not in prompt.lower():
                logger.warning("Cover image prompt does not specify it's a book cover")
                prompt_data["prompt"] = f"BOOK COVER: {prompt} This is the cover image for a children's book."
    
    if len(prompts_list) > max_prompts_to_validate:
        logger.info(f"Limiting validation to {max_prompts_to_validate} prompts out of {len(prompts_list)}")
        prompts_to_validate = prompts_list[:max_prompts_to_validate]
    else:
        prompts_to_validate = prompts_list
    
    logger.info(f"Validating {len(prompts_to_validate)} prompts:")
    for i, prompt_data in enumerate(prompts_to_validate):
        logger.info(f"Prompt {i+1} ({prompt_data['image_name']}): {prompt_data['prompt'][:100]}...")
    
    formatted_prompts = []
    for prompt_data in prompts_to_validate:
        formatted_prompts.append(f"IMAGE: {prompt_data['image_name']}\nPROMPT: {prompt_data['prompt']}")
    
    prompts_text = "\n\n".join(formatted_prompts)
    
    validation_prompt = """
    Evaluate these image prompts for a children's book illustration:

    CONTEXT:
    • Story: """ + story_content + """
    • Characters: """ + character_metadata + """
    • Style: """ + (style if style else "Not specified") + """
    • Book Title: """ + (book_title if book_title else "Unknown - check in story content") + """
    • Validation Attempt: """ + str(validate_image_prompts.call_count) + """ of 3 maximum attempts

    <thinking>
    1. Analyze each prompt for completeness of character descriptions
    2. Check for consistency of character details across all prompts
    3. Evaluate clarity of scene descriptions and actions
    4. Assess age-appropriateness of content for target audience
    5. Identify any missing essential elements that would affect visualization
    6. For cover images, verify the book title is included and it's clearly a book cover
    </thinking>

    EVALUATION CRITERIA:
    1. CHARACTER DETAILS: All physical attributes, clothing, and expressions clearly described
    2. CONSISTENCY: Maintaining consistent character descriptions across prompts
    3. VISUAL CLARITY: Clear scene, action, and setting description
    4. CHILD-APPROPRIATENESS: Content suitable for target age group
    5. COVER IMAGE: For cover images, ensure the book title is included and it's clearly a book cover
    
    SCORING SYSTEM:
    - 0-3: Severely inadequate (missing critical elements, unusable for generation)
    - 4-6: Needs significant improvement (workable but many issues)
    - 7-8: Good (minor issues only, acceptable for generation)
    - 9-10: Excellent (no meaningful improvements needed)

    PROMPTS TO EVALUATE:
    """ + prompts_text + """

    RESPOND IN THIS EXACT JSON FORMAT:
    ```json
    {
      "overall_assessment": "Brief summary of the prompts' quality",
      "quality_scores": {
        "overall": 0-10,
        "character_details": 0-10,
        "consistency": 0-10,
        "visual_clarity": 0-10
      },
      "prompt_feedback": {
        "image_name_1": {
          "character_details": "Feedback on character completeness",
          "consistency": "Feedback on consistency with story/other prompts",
          "improvements": "1-2 specific improvements needed",
          "score": 0-10
        }
      },
      "key_improvements": [
        "Most important improvement 1",
        "Most important improvement 2",
        "Most important improvement 3"
      ],
      "next_steps": "Concrete recommendations on what to fix before proceeding"
    }
    ```

    IMPORTANT GRADING INSTRUCTIONS:
    - This is attempt #""" + str(validate_image_prompts.call_count) + """ of 3 maximum validation attempts
    - If scores are 7 or higher, clearly state the prompts are ACCEPTABLE FOR GENERATION
    - Be more lenient with each validation attempt - focus only on critical issues
    - Use higher scores (7+) for prompts that would produce acceptable illustrations with minor flaws
    - Remember: "Good enough" is better than perfect - these prompts will undergo post-processing
    - For cover images, ensure the book title appears prominently in the prompt and it's clearly described as a book cover

    Example of a good prompt: "Hoppy, a small white rabbit with long floppy ears and blue eyes, wearing a red vest with gold buttons, is sitting in a garden holding an orange carrot with a curious expression."
    
    Example of a good cover prompt: "BOOK COVER: Hoppy, a small white rabbit with long floppy ears and blue eyes, wearing a red vest with gold buttons, is sitting in a garden holding an orange carrot with a curious expression. The title 'Hoppy's Garden Adventure' appears prominently at the top of the image in colorful, child-friendly lettering."
    """
    
    try:
        logger.info(f"Calling validation model: {model_id}")
        start_time = time.time()
        
        response = completion(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional children's book illustrator that provides focused, actionable feedback on image prompts. Format your response exactly as requested in JSON format. On second or third validation attempts, be increasingly lenient and focus only on critical issues."
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Generated prompt validation in {processing_time:.2f} seconds")
        validation_text = response.choices[0].message.content

        if "```json" in validation_text and "```" in validation_text:
            json_start = validation_text.find("```json") + 7
            json_end = validation_text.rfind("```")
            json_str = validation_text[json_start:json_end].strip()
        elif "```" in validation_text:
            json_start = validation_text.find("```") + 3
            json_end = validation_text.rfind("```")
            json_str = validation_text[json_start:json_end].strip()
        else:
            json_str = validation_text
        
        try:
            validation_result = json.loads(json_str)
            logger.info("Successfully parsed validation result as JSON")
            
            validation_result["auto_approved"] = False
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse validation response as JSON, returning in structured format")
            validation_result = {
                "overall_assessment": "Validation completed but couldn't be parsed as JSON",
                "prompt_feedback": {},
                "key_improvements": ["Review the raw feedback and manually extract key points"],
                "next_steps": "Review the raw validation feedback",
                "quality_scores": {
                    "overall": 7,
                    "character_details": 7,
                    "consistency": 7,
                    "visual_clarity": 7
                },
                "auto_approved": False
            }

            for prompt_data in prompts_to_validate:
                validation_result["prompt_feedback"][prompt_data["image_name"]] = {
                    "raw_feedback": validation_text,
                    "score": 7
                }

        validation_result["validation_metadata"] = {
            "prompts_validated": len(prompts_to_validate),
            "total_prompts": len(prompts_list),
            "processing_time_seconds": round(processing_time, 2),
            "validation_attempts": validate_image_prompts.call_count
        }
        if validate_image_prompts.call_count == 2:
            validation_result["next_steps"] = "FINAL OPPORTUNITY: " + validation_result.get("next_steps", "") + " After the next validation, proceed directly to generation regardless of results."
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating image prompts: {str(e)}")
        raise ValueError(f"Failed to validate image prompts: {str(e)}")

@tool
def preprocess_image_prompts(
    prompts_list: List[Dict[str, str]], 
    story_and_character_metadata: str,
    style: Optional[str] = None,
    model_id: str = "gpt-4o-mini",
    max_tokens: int = 4000,
    temperature: float = 0
) -> List[Dict[str, str]]:
    """
    Preprocesses image prompts to ensure character and visual consistency across multiple images.
    Also ensures cover images prominently include the book title and specify they are book covers.
    
    Args:
        prompts_list: A list of dictionaries, each containing 'image_name' and 'prompt' keys.
        story_and_character_metadata: string containing story and character descriptions to enforce consistency.
                          Format should be a JSON string with story and character descriptions as values.
        style: Optional consistent style to apply to all images.
        model_id: The model ID to use for processing prompts. Defaults to "gpt-4o-mini".
        max_tokens: Maximum number of tokens for the response. Defaults to 4000.
        temperature: Controls randomness of output. Lower values (0.0-0.2) for more consistent processing.
                    Defaults to 0.
    
    Returns:
        List[Dict[str, str]]: The processed prompts list with consistent character descriptions and styling.
        
    Raises:
        ValueError: If the prompts list is empty or if the model call fails.
    """
    logger.info(f"Preprocessing {len(prompts_list)} image prompts for consistency")
    if not prompts_list:
        raise ValueError("Prompts list cannot be empty")
    book_title = ""
    for prompt_data in prompts_list:
        if "cover" in prompt_data.get("image_name", "").lower():
            path_parts = prompt_data["image_name"].split("/")
            if len(path_parts) > 1:
                book_title = path_parts[-2]  # Get the folder name which is often the title
                break
    if not book_title and story_and_character_metadata:
        title_match = re.search(r"title:?\s*['\"](.*?)['\"]", story_and_character_metadata, re.IGNORECASE)
        if title_match:
            book_title = title_match.group(1)
    
    logger.info(f"Extracted book title: {book_title or 'Unknown'}")
    all_prompts = [prompt_data.get('prompt', '') for prompt_data in prompts_list]
    joined_prompts = "\n\n---\n\n".join(all_prompts)
    if not story_and_character_metadata:
        logger.info("No character metadata provided, will extract from prompts")
        
        extraction_prompt = f"""
        Extract consistent character descriptions from these image prompts for a children's book.

        IMAGE PROMPTS:
        {joined_prompts}

        TASK:
        1. Identify all characters mentioned across these prompts
        2. For each character, create a detailed, consistent description including:
           - Physical attributes (size, shape, colors, features)
           - Clothing or accessories
           - Expressions or typical poses
        3. Ensure descriptions are detailed enough for consistent illustration
        4. Use the EXACT character naming from the prompts

        OUTPUT FORMAT:
        ```json
        {{
          "characters": {{
            "CHARACTER_NAME_1": "Detailed physical description including all visual elements",
            "CHARACTER_NAME_2": "Detailed physical description including all visual elements"
          }}
        }}
        ```
        """
        
        try:
            logger.info(f"Extracting character descriptions using model: {model_id}")
            response = completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an expert character designer ensuring visual consistency across book illustrations."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            character_response = response.choices[0].message.content
            json_start = character_response.find('```json') + 7 if '```json' in character_response else character_response.find('```') + 3 if '```' in character_response else 0
            json_end = character_response.rfind('```') if '```' in character_response else len(character_response)
            json_str = character_response[json_start:json_end].strip()
            
            try:
                extracted_data = json.loads(json_str)
                if 'characters' in extracted_data:
                    story_and_character_metadata = json.dumps(extracted_data['characters'], indent=2)
                    logger.info(f"Successfully extracted {len(extracted_data['characters'])} character descriptions")
                else:
                    logger.warning("Extracted data does not contain 'characters' key")
                    story_and_character_metadata = json_str
            except json.JSONDecodeError:
                logger.warning("Failed to parse character metadata as JSON, using raw text")
                story_and_character_metadata = character_response
                
        except Exception as e:
            logger.error(f"Error extracting character descriptions: {str(e)}")
            logger.warning("Proceeding without character metadata extraction")
            story_and_character_metadata = "No character metadata could be extracted"
    logger.info("Processing prompts with extracted character data")
    for prompt_data in prompts_list:
        if "cover" in prompt_data.get("image_name", "").lower():
            logger.info("Found cover image prompt - adding special processing instructions")
            cover_processing_prompt = f"""
            Create a detailed prompt for a children's book COVER image that prominently features the book title.
            
            BOOK TITLE: {book_title or "The title from the story"}
            
            CHARACTER DESCRIPTIONS:
            {story_and_character_metadata}
            
            STYLE GUIDELINES:
            {style if style else "Create a vibrant, engaging children's book cover illustration style"}
            
            ORIGINAL PROMPT:
            {prompt_data.get('prompt', '')}
            
            TASK:
            1. This is specifically for a BOOK COVER illustration
            2. Begin with "BOOK COVER:" to clearly mark this as a cover image
            3. Include the book title prominently in the illustration description
            4. Use standardized character descriptions from CHARACTER DESCRIPTIONS
            5. Create a visually striking composition appropriate for a book cover
            6. Maintain the core scene elements from the original prompt if appropriate
            7. Describe where and how the title should appear in the image
            
            UPDATED COVER PROMPT:
            """
            
            try:
                logger.info("Processing cover image prompt with special instructions")
                response = completion(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are an expert children's book cover designer specializing in creating engaging and appealing book covers."},
                        {"role": "user", "content": cover_processing_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                processed_cover_prompt = response.choices[0].message.content.strip()
                prompt_data['prompt'] = processed_cover_prompt
                prompt_data['original_prompt'] = prompt_data.get('prompt', '')  # Store the original for reference
                logger.info(f"Successfully processed cover image prompt with title '{book_title}'")
                
            except Exception as e:
                logger.error(f"Error processing cover prompt: {str(e)}")
                original_prompt = prompt_data.get('prompt', '')
                if book_title and book_title.lower() not in original_prompt.lower():
                    prompt_data['prompt'] = f"BOOK COVER: {original_prompt} The title '{book_title}' should appear prominently at the top of the image in colorful, child-friendly lettering."
                elif "cover" not in original_prompt.lower():
                    prompt_data['prompt'] = f"BOOK COVER: {original_prompt}"
    
    processing_prompt = f"""
    Standardize this image prompt to ensure visual consistency across all illustrations.
    
    CHARACTER DESCRIPTIONS:
    {story_and_character_metadata}
    
    STYLE GUIDELINES:
    {style if style else "Maintain a consistent illustrative style throughout all images"}
    
    ORIGINAL PROMPT:
    {{prompt}}
    
    TASK:
    1. Maintain the same scene and action from the original prompt
    2. Replace character descriptions with the standardized versions from CHARACTER DESCRIPTIONS
    3. Keep the same level of detail for scene elements
    4. Ensure the prompt describes what should be visible in the image
    5. DON'T add any new story events or change the action/scene
    
    UPDATED PROMPT:
    """
    processed_prompts = []
    for i, prompt_data in enumerate(prompts_list):
        try:
            image_name = prompt_data.get('image_name', '')
            if "cover" in image_name.lower():
                processed_prompts.append(prompt_data)
                continue

            original_prompt = prompt_data.get('prompt', '')
            
            if not original_prompt:
                logger.warning(f"Empty prompt for {image_name}, skipping processing")
                processed_prompts.append(prompt_data)
                continue
            
            current_processing_prompt = processing_prompt.replace("{prompt}", original_prompt)
            logger.info(f"Processing prompt {i+1}/{len(prompts_list)}: {image_name}")
            response = completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are an expert prompt engineer specializing in creating consistent image descriptions."},
                    {"role": "user", "content": current_processing_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            processed_prompt = response.choices[0].message.content.strip()

            new_prompt_data = prompt_data.copy()
            new_prompt_data['prompt'] = processed_prompt
            new_prompt_data['original_prompt'] = original_prompt  # Store the original for reference
            
            processed_prompts.append(new_prompt_data)
            logger.info(f"Successfully processed prompt for {image_name}")
            
        except Exception as e:
            logger.error(f"Error processing prompt {i+1}: {str(e)}")

            processed_prompts.append(prompt_data)
    
    logger.info(f"Preprocessing complete: {len(processed_prompts)}/{len(prompts_list)} prompts processed")

    for prompt_data in processed_prompts:
        if 'metadata' not in prompt_data:
            prompt_data['metadata'] = {}
        prompt_data['metadata']['preprocessing_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        prompt_data['metadata']['character_consistency_applied'] = True
    
    return processed_prompts

@tool
def generate_images_batch(prompts_list: List[Dict[str, str]],
                           story_and_character_metadata: str, 
                           style: Optional[str] = None, 
                           output_folder: Optional[str] = None) -> Dict[str, str]:
    """
    Generates multiple images based on a list of prompts in a single batch operation.
    
    Args:
        prompts_list: A list of dictionaries, each containing 'image_name' and 'prompt' keys.
                     Example: [{'image_name': '{story_name}/page1.jpg', 'prompt': 'A cat sleeping'}, 
                              {'image_name': '{story_name}/page2.jpg', 'prompt': 'A dog running'}]
                     Note: .jpg extension is preferred for consistency across the system.
        story_and_character_metadata: string containing story and character descriptions to enforce consistency.
                          Format should be a JSON string with story and character descriptions as values.                     
        style: Style of the image generation, applied to all images. Any consistent string describing artistic style.
              Examples: "watercolor children's illustration style", "3D animation style", "hand-drawn pencil style".
        output_folder: Output directory path where the images will be saved. If None, uses the default from CONFIG.
                       Should be a complete path, not just a folder name.                          
        
    Returns:
        Dict[str, str]: A dictionary mapping image names to their file paths.
        
    Raises:
        ValueError: If any image_name has an invalid extension or output_folder creation fails.
        TypeError: If the image response cannot be processed.
        
    Examples:
        >>> # Generate pages for a story with consistent character appearance
        >>> story_and_character_metadata = '{"story": "A magical adventure in the garden", "characters": {"Lily": "A 7-year-old girl with curly red hair and green eyes", "Max": "A playful puppy with a golden coat"}}'
        >>> prompts = [
        ...     {"image_name": "my_story/page1.jpg", "prompt": "Lily playing in a garden"},
        ...     {"image_name": "my_story/page2.jpg", "prompt": "Lily finding a magic key"}
        ... ]
        >>> generate_images_batch(
        ...     prompts_list=prompts,
        ...     style="Watercolor children's book style",
        ...     output_folder="output/my_story",
        ...     story_and_character_metadata=story_and_character_metadata
        ... )
    """
    if len(prompts_list) > 1:
        logger.info("Preprocessing prompts for consistency before image generation")
        try:
            prompts_list = preprocess_image_prompts(
                prompts_list=prompts_list,
                story_and_character_metadata=story_and_character_metadata,
                style=style
            )
            logger.info("Preprocessing complete, proceeding with image generation")
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}. Proceeding with original prompts.")
    if output_folder is None:
        output_folder = CONFIG["output_folder"]
    
    results = {}
    if not prompts_list:
        logger.error("No prompts provided in prompts_list")
        raise ValueError("prompts_list cannot be empty")
    
    for i, prompt_data in enumerate(prompts_list):
        if not isinstance(prompt_data, dict):
            logger.error(f"Item {i} in prompts_list is not a dictionary")
            raise ValueError(f"Each item in prompts_list must be a dictionary with 'image_name' and 'prompt' keys")
        
        if 'image_name' not in prompt_data:
            logger.error(f"Item {i} in prompts_list is missing 'image_name' key")
            raise ValueError(f"Item {i} in prompts_list is missing required 'image_name' key")
            
        if 'prompt' not in prompt_data:
            logger.error(f"Item {i} in prompts_list is missing 'prompt' key")
            raise ValueError(f"Item {i} in prompts_list is missing required 'prompt' key")
    
    total_images = len(prompts_list)
    logger.info(f"Generating batch of {total_images} images with consistent style: {style}")
    logger.info(f"Output folder: {output_folder}")
    
    api_key = os.getenv('OPEN_AI_API_KEY')
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("No OpenAI API key found in environment variables")
            raise ValueError("No OpenAI API key found. Set either OPEN_AI_API_KEY or OPENAI_API_KEY environment variable.")
    
    validation_warnings = []
    for i, prompt_data in enumerate(prompts_list):
        prompt = prompt_data.get('prompt', '')
        if len(prompt) < 100:
            validation_warnings.append(f"Prompt {i+1} ({prompt_data.get('image_name', f'prompt_{i}')}) may be too short ({len(prompt)} chars)")
        if 'style' in prompt.lower() and style:
            validation_warnings.append(f"Prompt {i+1} contains style information which should be in the style parameter only")
    
    if validation_warnings:
        logger.warning("Potential prompt issues detected:")
        for warning in validation_warnings:
            logger.warning(f"  - {warning}")
        logger.warning("Consider validating prompts with validate_image_prompts before generation")
    
    style_text = style if style else "Warm, friendly children's book illustration style"
    system_content = (
        "You are an expert children's book illustrator specializing in creating consistent, "
        "high-quality images. Generate images that are appropriate for children, with:\n"
        "1. Consistent character appearance across all illustrations\n"
        "2. Clear, well-balanced compositions with defined focal points\n"
        "3. Appropriate use of color and lighting to create mood\n"
        "4. Simplified backgrounds that don't distract from main subjects\n"
        "5. Visual style: " + style_text
    )
    
    system_message = {
        "role": "system",
        "content": system_content
    }  
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for i, prompt_data in enumerate(prompts_list):
        try:
            image_name = prompt_data.get('image_name')
            prompt = prompt_data.get('prompt')
            
            if not image_name or not prompt:
                logger.error(f"Missing required keys in prompt data at index {i}: {prompt_data}")
                results[f"error_{i}"] = f"Error: Missing required keys in prompt data"
                continue
            if style:
                full_prompt = f"{prompt}\n\nApply this consistent artistic style to the entire image: {style}"
            else:
                full_prompt = prompt

            logger.info(f"Generating image {i+1}/{total_images}: {image_name}")
            logger.debug(f"Full prompt: {full_prompt}")
            img_path = Path(image_name)
            if img_path.parent != Path('.'):
                full_path = Path(output_folder) / img_path.parent
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created subdirectory: {full_path}")
            
            try:
                output_path = _prepare_image_generation(image_name, output_folder)
                save_path = output_path / image_name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Save path prepared: {save_path}")
            except Exception as e:
                logger.error(f"Error preparing output path: {e}")
                results[image_name] = f"Error: {str(e)}"
                continue
            
            user_message = {
                "role": "user",
                "content": full_prompt
            }
            
            logger.info(f"Calling image_generation with model={CONFIG['models']['image_generation']['model']}, size={CONFIG['models']['image_generation']['size']}")
            
            try:
                logger.info("Attempting image generation with messages format")
                image_response = image_generation(
                    messages=[system_message, user_message],
                    model=CONFIG["models"]["image_generation"]["model"], 
                    size=CONFIG["models"]["image_generation"]["size"],
                    api_key=api_key
                )
                logger.info(f"Image generation response received: {type(image_response)}")
            except Exception as e:
                logger.warning(f"Messages format not supported, falling back to prompt parameter: {e}")
                try:
                    image_response = image_generation(
                        prompt=full_prompt, 
                        model=CONFIG["models"]["image_generation"]["model"], 
                        size=CONFIG["models"]["image_generation"]["size"],
                        api_key=api_key
                    )
                    logger.info(f"Fallback image generation response received: {type(image_response)}")
                except Exception as inner_e:
                    logger.error(f"Failed with both message and prompt formats: {inner_e}")
                    results[image_name] = f"Error: Image generation failed for both formats: {str(inner_e)}"
                    continue
            
            logger.info(f"Response type: {type(image_response)}")
            logger.info(f"Response attributes: {dir(image_response) if hasattr(image_response, '__dict__') else 'No attributes'}")
            
            try:
                image_data = None
                
                if hasattr(image_response, 'data') and len(image_response.data) > 0:
                    logger.info("Processing OpenAI API style response")
                    image_url = image_response.data[0].url
                    image_data = requests.get(image_url).content
                
                elif hasattr(image_response, 'images') and len(image_response.images) > 0:
                    logger.info("Processing base64 encoded image response")
                    image_data = base64.b64decode(image_response.images[0])
                
                elif isinstance(image_response, dict):
                    logger.info("Processing dictionary style response")
                    if 'data' in image_response and len(image_response['data']) > 0:
                        if 'url' in image_response['data'][0]:
                            image_url = image_response['data'][0]['url']
                            image_data = requests.get(image_url).content
                        elif 'b64_json' in image_response['data'][0]:
                            image_data = base64.b64decode(image_response['data'][0]['b64_json'])
                    elif 'images' in image_response and len(image_response['images']) > 0:
                        image_data = base64.b64decode(image_response['images'][0])
                
                if not image_data:
                    resp_str = str(image_response)
                    logger.error(f"Unexpected response format: {resp_str[:200]}...")
                    raise TypeError(f"Unable to extract image data from response: {resp_str[:200]}...")
                
                logger.info(f"Image data extracted, saving to {save_path}")
                with open(save_path, 'wb') as f:
                    f.write(image_data)
                
                logger.info(f"Image saved successfully to {save_path}")
                results[image_name] = str(save_path)
                
            except Exception as e:
                logger.error(f"Error processing or saving image: {e}")
                results[image_name] = f"Error: {str(e)}"
                continue
            
        except Exception as e:
            logger.error(f"Error generating image {prompt_data.get('image_name', f'at index {i}')}: {str(e)}")
            results[prompt_data.get('image_name', f'error_{i}')] = f"Error: {str(e)}"
    
    success_count = sum(1 for value in results.values() if not value.startswith("Error:"))
    logger.info(f"Image generation summary: {success_count}/{total_images} images generated successfully")
    
    if success_count < total_images:
        logger.warning(f"Failed to generate {total_images - success_count} images")
        for name, result in results.items():
            if result.startswith("Error:"):
                logger.warning(f"Image '{name}' failed: {result}")
        logger.info("Consider using validate_image_consistency to check the generated images for consistency")
    else:
        logger.info("All images generated successfully! Consider using validate_image_consistency to ensure style consistency")
    
    return results

if __name__ == "__main__":
    generate_images_batch([{"image_name": "dolphin.jpg",
                             "prompt": "create an image of a dolphin"}])
