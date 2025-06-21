# Windows Screenshot MCP Server

A Windows-only Model Context Protocol (MCP) server that enables AI agents to capture screenshots of applications or the entire screen, with optional visual question answering through local or remote AI models.

## Features

- **Full Screen Capture**: Capture screenshots of the entire screen or specific monitors
- **Window-Specific Capture**: Target specific application windows by title or process name
- **Window Enumeration**: List all visible windows with their process information
- **AI-Powered Analysis**: Analyze screenshots using OpenAI, Anthropic Claude, or local models
- **Multiple Image Formats**: Support for PNG and JPEG output
- **Windows Integration**: Deep Windows API integration for reliable window targeting

## Requirements

- Windows operating system (Windows 10/11 recommended)
- Python 3.8 or higher
- Required Python packages (see requirements.txt)

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys (optional, for AI analysis):
   ```bash
   # For OpenAI
   set OPENAI_API_KEY=your_openai_api_key
   
   # For Anthropic Claude
   set ANTHROPIC_API_KEY=your_anthropic_api_key
   
   # For local models (e.g., Ollama)
   set LOCAL_MODEL_URL=http://localhost:11434
   ```

## Usage

Run the MCP server:
```bash
python screenshot_server.py
```

## Available Tools

### `capture_screen`
Capture a screenshot of the entire screen or a specific monitor.

**Parameters:**
- `monitor` (integer, optional): Monitor number (0 for primary, 1+ for additional)
- `format` (string, optional): Image format ("png" or "jpeg", default: "png")

### `capture_window`
Capture a screenshot of a specific application window.

**Parameters:**
- `window_title` (string): Title or partial title of the window
- `process_name` (string): Process name (e.g., "notepad.exe")
- `format` (string, optional): Image format ("png" or "jpeg", default: "png")

Note: Either `window_title` or `process_name` is required.

### `list_windows`
List all visible windows with their titles and process names.

**Parameters:** None

### `analyze_screenshot`
Analyze a screenshot using AI and answer questions about it.

**Parameters:**
- `image_data` (string): Base64 encoded image data
- `question` (string): Question to ask about the image
- `model_provider` (string, optional): "openai", "anthropic", or "local" (default: "openai")
- `model_name` (string, optional): Specific model name (default: "gpt-4-vision-preview")

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4 Vision analysis
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude analysis
- `LOCAL_MODEL_URL`: URL for local model API (default: http://localhost:11434)

### Supported AI Models

#### OpenAI
- `gpt-4-vision-preview`
- `gpt-4o`
- `gpt-4o-mini`

#### Anthropic
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-3-opus-20240229`

#### Local Models
- Any Ollama model with vision capabilities (e.g., `llava`, `bakllava`)
- Custom local vision models with compatible API

## Examples

### Basic Screenshot Capture
```python
# Capture entire screen
{"tool": "capture_screen", "arguments": {}}

# Capture specific monitor
{"tool": "capture_screen", "arguments": {"monitor": 1, "format": "jpeg"}}
```

### Window-Specific Capture
```python
# Capture by window title
{"tool": "capture_window", "arguments": {"window_title": "Notepad"}}

# Capture by process name
{"tool": "capture_window", "arguments": {"process_name": "chrome.exe"}}
```

### AI Analysis
```python
# Analyze with OpenAI
{
  "tool": "analyze_screenshot",
  "arguments": {
    "image_data": "base64_encoded_image_data",
    "question": "What applications are visible in this screenshot?",
    "model_provider": "openai"
  }
}

# Analyze with local model
{
  "tool": "analyze_screenshot",
  "arguments": {
    "image_data": "base64_encoded_image_data",
    "question": "Describe what you see in this image",
    "model_provider": "local",
    "model_name": "llava"
  }
}
```

## Security Considerations

- This server requires Windows API access and can capture sensitive information
- Screenshots may contain private data - ensure proper handling
- API keys should be stored securely and not committed to version control
- Consider network security when using remote AI models

## Troubleshooting

### Common Issues

1. **Import Error for Windows modules**: Ensure you're running on Windows
2. **Permission denied**: Run as administrator if capturing system windows
3. **Window not found**: Check window titles with `list_windows` tool first
4. **AI analysis fails**: Verify API keys are set correctly

### Debug Mode

Enable debug logging by modifying the logging level in the script:
```python
logging.basicConfig(level=logging.DEBUG)
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Windows
5. Submit a pull request