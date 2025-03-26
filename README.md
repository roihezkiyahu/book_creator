# Book Creator

A powerful tool for automatically generating illustrated children's books using AI. This project orchestrates multiple specialized agents to create complete books from simple prompts.

## Features

- 📚 Story Generation: Creates age-appropriate, engaging stories
- 🎨 Image Generation: Produces illustrations in customizable artistic styles
- 📄 Layout Creation: Combines text and images into cohesive book layouts
- 🤖 Multi-Agent System: Uses specialized agents for each part of the creation process

## Prerequisites

- Python 3.8+
- OpenAI API key (for GPT-4o-mini and DALL-E-3 models)
- Hugging Face API token (optional, for alternative models)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/roihezkiyahu/book_creator.git
   cd book_creator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   If no requirements.txt is present, install these essential packages:
   ```
   pip install python-dotenv smolagents pillow
   ```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following credentials:

```
# OpenAI API Key (required)
OPEN_AI_API_KEY=your_openai_api_key_here
```

### Configuration Settings

The project uses a configuration file (`config.py`) to manage settings like:
- Model selection
- Output folder paths
- Maximum steps for each agent
- Verbosity levels

You can modify these settings to customize the book creation process.

## Usage

### Basic Example

```python
from book_creator import create_book

result = create_book(
    title="Geffen's Jungle Adventure",
    story_description="A story about a curious child named Geffen who goes on an adventure to meet different jungle animals in their natural habitats.",
    num_pages=3,
    lines_per_page=2,
    artistic_style="watercolor children's digital illustration style with soft colors and simple shapes"
)
```

### Parameters

- `title`: The title of your book
- `story_description`: A brief description of the story's content
- `num_pages`: Number of pages in the book (default: 3)
- `lines_per_page`: Number of text lines per page (default: 4)
- `artistic_style`: The visual style for illustrations (default: "watercolor")

### Output

The generated book assets will be saved in the configured output directory (`book_creator_output` by default):

```
book_creator_output/
└── YourBookTitle/
    ├── story_details.txt       # Complete story content
    ├── images/                 # Raw illustrations
    │   ├── cover.jpg
    │   ├── page1.jpg
    │   └── ...
    └── text_adjacent/          # Final layouts with text and images
        ├── cover.jpg
        ├── page1.jpg
        └── ...
```

### Example Book: Geffen's Jungle Adventure

Below are actual output images from a book generated using Book Creator:

#### Cover
![Geffen's Jungle Adventure - Cover](Geffen\'s%20Jungle%20Adventure/text_adjacent/cover.jpg)

#### Page 1 (Text Adjacent Layout)
![Geffen's Jungle Adventure - Page 1](Geffen\'s%20Jungle%20Adventure/text_adjacent/page1.jpg)

#### Page 2 (Text Adjacent Layout)
![Geffen's Jungle Adventure - Page 2](Geffen\'s%20Jungle%20Adventure/text_adjacent/page2.jpg)

#### Page 3 (Text Adjacent Layout)
![Geffen's Jungle Adventure - Page 3](Geffen\'s%20Jungle%20Adventure/text_adjacent/page3.jpg)

The Book Creator generates both text-adjacent layouts (shown above) and text-overlay layouts. The text-adjacent layout positions text beside illustrations for optimal readability, while the overlay layout places text directly on the illustrations.

## Project Structure

- `book_creator.py`: Main entry point and orchestration
- `agent_factory.py`: Creates and configures the specialized agents
- `utils.py`: Utility functions for directory setup and prompt preparation
- `config.py`: Configuration settings
- `tools/`: Custom tools used by the agents
- `prompts/`: Template prompts for different stages of book creation

## Troubleshooting

- **API Key Issues**: Ensure your API keys are properly set in the `.env` file
- **Image Generation Errors**: Check that your DALL-E API access is enabled in your OpenAI account
- **Output Directory**: Make sure the configured output directory is writable

## Acknowledgements

- This project uses [smolagents](https://github.com/smol-ai/smolagents) for agent coordination
- Image generation powered by DALL-E 3
- Text generation powered by GPT-4o-mini
