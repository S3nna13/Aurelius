"""OpenAI-compatible Python client for testing Aurelius.

Supports streaming and non-streaming modes, system prompts,
and multi-turn conversation history. Works with both Ollama
and SGLang endpoints.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Message:
    """A single conversation message."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatClient:
    """OpenAI-compatible chat client for Aurelius endpoints.

    Works with any OpenAI-compatible API server including
    Ollama (port 11434) and SGLang (port 8000).
    """

    base_url: str = "http://localhost:11434"
    """Base URL of the inference server."""

    model: str = "aurelius"
    """Model name to use in API calls."""

    system_prompt: str = "You are Aurelius, a helpful, harmless, and honest AI assistant."

    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout: float = 120.0

    history: list[Message] = field(default_factory=list)
    """Conversation history for multi-turn interactions."""

    def __post_init__(self) -> None:
        # Normalize base URL
        self.base_url = self.base_url.rstrip("/")

    @property
    def _chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _build_messages(self, user_message: str) -> list[dict[str, str]]:
        """Build the full message list including system prompt and history."""
        messages: list[dict[str, str]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self.history:
            messages.append(msg.to_dict())

        messages.append({"role": "user", "content": user_message})
        return messages

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the API request payload."""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    def chat(self, user_message: str) -> str:
        """Send a message and get a complete response (non-streaming).

        Args:
            user_message: The user's input text.

        Returns:
            The assistant's response text.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
            httpx.ConnectError: If the server is unreachable.
        """
        messages = self._build_messages(user_message)
        payload = self._build_payload(messages, stream=False)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self._chat_url, json=payload)
            response.raise_for_status()

        data = response.json()
        assistant_content = data["choices"][0]["message"]["content"]

        # Update conversation history
        self.history.append(Message(role="user", content=user_message))
        self.history.append(Message(role="assistant", content=assistant_content))

        return assistant_content

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """Send a message and stream the response token by token.

        Args:
            user_message: The user's input text.

        Yields:
            Response text chunks as they arrive.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses.
            httpx.ConnectError: If the server is unreachable.
        """
        messages = self._build_messages(user_message)
        payload = self._build_payload(messages, stream=True)

        full_response: list[str] = []

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self._chat_url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue

                    data_str = line[len("data: ") :]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_response.append(content)
                        yield content

        # Update conversation history
        assistant_content = "".join(full_response)
        self.history.append(Message(role="user", content=user_message))
        self.history.append(Message(role="assistant", content=assistant_content))

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self.history.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt.

        Args:
            prompt: New system prompt text.
        """
        self.system_prompt = prompt

    def health_check(self) -> bool:
        """Check if the inference server is reachable.

        Returns:
            True if the server responds to a health check.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except httpx.ConnectError:
            return False


def interactive_session(client: ChatClient, *, stream: bool = True) -> None:
    """Run an interactive chat session in the terminal.

    Args:
        client: Configured ChatClient instance.
        stream: Whether to stream responses.
    """
    print(f"Aurelius Chat Client - connected to {client.base_url}")
    print(f"Model: {client.model}")
    print("Commands: /clear (reset history), /system <prompt>, /quit")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye.")
            break

        if user_input == "/clear":
            client.clear_history()
            print("[History cleared]")
            continue

        if user_input.startswith("/system "):
            new_prompt = user_input[len("/system ") :]
            client.set_system_prompt(new_prompt)
            print(f"[System prompt updated: {new_prompt[:80]}...]")
            continue

        print("\nAurelius: ", end="", flush=True)

        try:
            if stream:
                for chunk in client.chat_stream(user_input):
                    print(chunk, end="", flush=True)
                print()
            else:
                response = client.chat(user_input)
                print(response)
        except httpx.ConnectError:
            print("[Error: Cannot connect to server. Is it running?]")
        except httpx.HTTPStatusError as exc:
            print(f"[Error: HTTP {exc.response.status_code}]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aurelius Chat Client")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Server base URL (default: http://localhost:11434 for Ollama)",
    )
    parser.add_argument(
        "--model",
        default="aurelius",
        help="Model name (default: aurelius)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming mode",
    )
    parser.add_argument(
        "--sglang",
        action="store_true",
        help="Use SGLang endpoint (localhost:8000)",
    )
    parser.add_argument(
        "--message",
        "-m",
        help="Send a single message and exit (non-interactive)",
    )
    args = parser.parse_args()

    base_url = args.base_url
    if args.sglang:
        base_url = "http://localhost:8000"

    chat_client = ChatClient(base_url=base_url, model=args.model)

    if args.message:
        # Single-shot mode
        if args.no_stream:
            print(chat_client.chat(args.message))
        else:
            for token in chat_client.chat_stream(args.message):
                sys.stdout.write(token)
                sys.stdout.flush()
            print()
    else:
        interactive_session(chat_client, stream=not args.no_stream)
