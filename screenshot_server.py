#!/usr/bin/env python3
"""
Windows Screenshot MCP Server
Enables AI agents to capture screenshots with optional visual question answering.
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import requests
from PIL import Image, ImageGrab
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Windows-specific imports
if sys.platform != "win32":
    raise RuntimeError("This MCP server is designed to run only on Windows")

import win32gui
import win32process
import win32api
import win32con

logger = logging.getLogger(__name__)

class WindowsScreenshotServer:
    def __init__(self):
        self.server = Server("windows-screenshot")
        self.ai_config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "local_model_url": os.getenv("LOCAL_MODEL_URL", "http://localhost:11434"),
        }
        
        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="capture_screen",
                    description="Capture a screenshot of the entire screen or a specific monitor",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "monitor": {
                                "type": "integer",
                                "description": "Monitor number to capture (0 for primary, 1+ for additional monitors)",
                                "default": 0
                            },
                            "format": {
                                "type": "string",
                                "enum": ["png", "jpeg"],
                                "description": "Image format for the screenshot",
                                "default": "png"
                            }
                        }
                    }
                ),
                Tool(
                    name="capture_window",
                    description="Capture a screenshot of a specific application window",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "window_title": {
                                "type": "string",
                                "description": "Title or partial title of the window to capture"
                            },
                            "process_name": {
                                "type": "string",
                                "description": "Process name of the application (e.g., 'notepad.exe')"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["png", "jpeg"],
                                "description": "Image format for the screenshot",
                                "default": "png"
                            }
                        },
                        "oneOf": [
                            {"required": ["window_title"]},
                            {"required": ["process_name"]}
                        ]
                    }
                ),
                Tool(
                    name="list_windows",
                    description="List all visible windows with their titles and process names",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="analyze_screenshot",
                    description="Analyze a screenshot using AI and answer questions about it",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_data": {
                                "type": "string",
                                "description": "Base64 encoded image data"
                            },
                            "question": {
                                "type": "string",
                                "description": "Question to ask about the image"
                            },
                            "model_provider": {
                                "type": "string",
                                "enum": ["openai", "anthropic", "local"],
                                "description": "AI model provider to use",
                                "default": "openai"
                            },
                            "model_name": {
                                "type": "string",
                                "description": "Specific model name (e.g., 'gpt-4-vision-preview', 'claude-3-sonnet')",
                                "default": "gpt-4-vision-preview"
                            }
                        },
                        "required": ["image_data", "question"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent]:
            try:
                if name == "capture_screen":
                    return await self._capture_screen(**arguments)
                elif name == "capture_window":
                    return await self._capture_window(**arguments)
                elif name == "list_windows":
                    return await self._list_windows()
                elif name == "analyze_screenshot":
                    return await self._analyze_screenshot(**arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _capture_screen(self, monitor: int = 0, format: str = "png") -> List[TextContent | ImageContent]:
        """Capture screenshot of entire screen or specific monitor"""
        try:
            # Get all monitors
            monitors = win32api.EnumDisplayMonitors()
            
            if monitor >= len(monitors):
                return [TextContent(type="text", text=f"Monitor {monitor} not found. Available monitors: 0-{len(monitors)-1}")]
            
            if monitor == 0:
                # Capture entire screen
                screenshot = ImageGrab.grab()
            else:
                # Capture specific monitor
                monitor_info = monitors[monitor]
                bbox = monitor_info[2]  # (left, top, right, bottom)
                screenshot = ImageGrab.grab(bbox=bbox)
            
            # Convert to requested format
            img_buffer = io.BytesIO()
            screenshot.save(img_buffer, format=format.upper())
            img_data = base64.b64encode(img_buffer.getvalue()).decode()
            
            return [
                TextContent(type="text", text=f"Screenshot captured successfully from monitor {monitor}"),
                ImageContent(
                    type="image",
                    data=img_data,
                    mimeType=f"image/{format.lower()}"
                )
            ]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Failed to capture screen: {str(e)}")]

    async def _capture_window(self, window_title: Optional[str] = None, 
                            process_name: Optional[str] = None, format: str = "png") -> List[TextContent | ImageContent]:
        """Capture screenshot of specific window"""
        try:
            hwnd = None
            
            if window_title:
                # Find window by title
                def enum_windows_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if window_title.lower() in title.lower():
                            windows.append((hwnd, title))
                    return True
                
                windows = []
                win32gui.EnumWindows(enum_windows_callback, windows)
                
                if not windows:
                    return [TextContent(type="text", text=f"No window found with title containing '{window_title}'")]
                
                hwnd = windows[0][0]  # Use first match
                
            elif process_name:
                # Find window by process name
                def enum_windows_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        try:
                            process = psutil.Process(pid)
                            if process.name().lower() == process_name.lower():
                                windows.append((hwnd, win32gui.GetWindowText(hwnd)))
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    return True
                
                windows = []
                win32gui.EnumWindows(enum_windows_callback, windows)
                
                if not windows:
                    return [TextContent(type="text", text=f"No window found for process '{process_name}'")]
                
                hwnd = windows[0][0]  # Use first match
            
            if not hwnd:
                return [TextContent(type="text", text="No window handle found")]
            
            # Get window rectangle
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x1, y1 = rect
            
            # Bring window to front
            win32gui.SetForegroundWindow(hwnd)
            await asyncio.sleep(0.1)  # Small delay to ensure window is in front
            
            # Capture window
            screenshot = ImageGrab.grab(bbox=(x, y, x1, y1))
            
            # Convert to requested format
            img_buffer = io.BytesIO()
            screenshot.save(img_buffer, format=format.upper())
            img_data = base64.b64encode(img_buffer.getvalue()).decode()
            
            window_info = win32gui.GetWindowText(hwnd)
            
            return [
                TextContent(type="text", text=f"Window screenshot captured: {window_info}"),
                ImageContent(
                    type="image",
                    data=img_data,
                    mimeType=f"image/{format.lower()}"
                )
            ]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Failed to capture window: {str(e)}")]

    async def _list_windows(self) -> List[TextContent]:
        """List all visible windows"""
        try:
            windows = []
            
            def enum_windows_callback(hwnd, windows_list):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title.strip():  # Only include windows with titles
                        try:
                            _, pid = win32process.GetWindowThreadProcessId(hwnd)
                            process = psutil.Process(pid)
                            process_name = process.name()
                            windows_list.append({
                                "title": title,
                                "process": process_name,
                                "pid": pid,
                                "hwnd": hwnd
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            windows_list.append({
                                "title": title,
                                "process": "Unknown",
                                "pid": pid,
                                "hwnd": hwnd
                            })
                return True
            
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            # Format output
            output = "Visible Windows:\n"
            for i, window in enumerate(windows, 1):
                output += f"{i}. {window['title']} ({window['process']}, PID: {window['pid']})\n"
            
            return [TextContent(type="text", text=output)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Failed to list windows: {str(e)}")]

    async def _analyze_screenshot(self, image_data: str, question: str, 
                                model_provider: str = "openai", model_name: str = "gpt-4-vision-preview") -> List[TextContent]:
        """Analyze screenshot using AI models"""
        try:
            if model_provider == "openai":
                return await self._analyze_with_openai(image_data, question, model_name)
            elif model_provider == "anthropic":
                return await self._analyze_with_anthropic(image_data, question, model_name)
            elif model_provider == "local":
                return await self._analyze_with_local_model(image_data, question, model_name)
            else:
                return [TextContent(type="text", text=f"Unsupported model provider: {model_provider}")]
                
        except Exception as e:
            return [TextContent(type="text", text=f"Failed to analyze screenshot: {str(e)}")]

    async def _analyze_with_openai(self, image_data: str, question: str, model_name: str) -> List[TextContent]:
        """Analyze image using OpenAI's vision models"""
        if not self.ai_config["openai_api_key"]:
            return [TextContent(type="text", text="OpenAI API key not configured")]
        
        headers = {
            "Authorization": f"Bearer {self.ai_config['openai_api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        return [TextContent(type="text", text=f"OpenAI Analysis: {answer}")]

    async def _analyze_with_anthropic(self, image_data: str, question: str, model_name: str) -> List[TextContent]:
        """Analyze image using Anthropic's Claude models"""
        if not self.ai_config["anthropic_api_key"]:
            return [TextContent(type="text", text="Anthropic API key not configured")]
        
        headers = {
            "x-api-key": self.ai_config["anthropic_api_key"],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": model_name,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post("https://api.anthropic.com/v1/messages",
                               headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        answer = result["content"][0]["text"]
        
        return [TextContent(type="text", text=f"Claude Analysis: {answer}")]

    async def _analyze_with_local_model(self, image_data: str, question: str, model_name: str) -> List[TextContent]:
        """Analyze image using local model (e.g., Ollama with vision models)"""
        try:
            # Assuming Ollama-style API
            payload = {
                "model": model_name,
                "prompt": question,
                "images": [image_data],
                "stream": False
            }
            
            response = requests.post(f"{self.ai_config['local_model_url']}/api/generate",
                                   json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "No response from local model")
            
            return [TextContent(type="text", text=f"Local Model Analysis: {answer}")]
            
        except Exception as e:
            return [TextContent(type="text", text=f"Local model error: {str(e)}")]

async def main():
    """Main entry point"""
    if sys.platform != "win32":
        print("This MCP server is designed to run only on Windows", file=sys.stderr)
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run server
    screenshot_server = WindowsScreenshotServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await screenshot_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="windows-screenshot",
                server_version="1.0.0",
                capabilities=screenshot_server.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())