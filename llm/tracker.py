"""Token usage, cost calculation, and named session tracking for LLM API calls."""

import atexit
from datetime import datetime
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MD_FILENAME = "session_token_costs.md"

# Model pricing table: rates per 1,000,000 tokens USD (input_rate_per_1M, output_rate_per_1M)
MODEL_PRICING: Dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
}
DEFAULT_PRICING = (0.15, 0.60)


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string using 4 chars per token rule of thumb."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def calculate_call_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the USD cost of an LLM API call based on token counts and model pricing."""
    model_key = model_name.lower().strip()
    input_rate, output_rate = DEFAULT_PRICING
    for name, rates in MODEL_PRICING.items():
        if name in model_key or model_key in name:
            input_rate, output_rate = rates
            break

    input_cost = (prompt_tokens / 1_000_000.0) * input_rate
    output_cost = (completion_tokens / 1_000_000.0) * output_rate
    return input_cost + output_cost


class TokenCostTracker:
    """Tracks token usage, API call history, costs across LLM invocations, and session summaries."""

    def __init__(self) -> None:
        self.call_history: list[dict] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

        # Session tracking state
        self.session_name: Optional[str] = None
        self.session_start_time: Optional[str] = None
        self.session_calls: list[dict] = []

    def start_session(self, session_name: Optional[str] = None) -> None:
        """Start a new session for grouping LLM calls under a named task.

        Args:
            session_name: Human-readable task description (e.g. 'Exam Generation (Physics Edexcel 20 Marks)').
        """
        if self.session_calls:
            # End previous session before starting a new one
            self.end_session()

        self.session_name = session_name or "LLM General Session"
        self.session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_calls = []
        logger.info("Started LLM tracking session: '%s'", self.session_name)

    def track_call(
        self,
        model_name: str,
        prompt: str,
        response_obj_or_text: Any,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> dict:
        """Record an LLM call, calculate tokens/costs, print cost outline, and return summary dict."""
        if not self.session_start_time:
            self.start_session("LLM General Session")

        input_tokens = prompt_tokens
        output_tokens = completion_tokens

        # Check for LangChain AIMessage usage_metadata
        if input_tokens is None or output_tokens is None:
            usage = getattr(response_obj_or_text, "usage_metadata", None)
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")

        # Check for response_metadata token_usage
        if input_tokens is None or output_tokens is None:
            meta = getattr(response_obj_or_text, "response_metadata", {})
            if isinstance(meta, dict):
                token_usage = meta.get("token_usage", {}) or meta.get("usage", {})
                if isinstance(token_usage, dict):
                    input_tokens = input_tokens or token_usage.get("prompt_tokens")
                    output_tokens = output_tokens or token_usage.get("completion_tokens")

        # Fallback estimation
        if input_tokens is None:
            input_tokens = estimate_tokens(str(prompt))
        if output_tokens is None:
            text = getattr(response_obj_or_text, "content", response_obj_or_text)
            output_tokens = estimate_tokens(str(text))

        total_tokens = input_tokens + output_tokens
        call_cost = calculate_call_cost(model_name, input_tokens, output_tokens)

        self.total_prompt_tokens += input_tokens
        self.total_completion_tokens += output_tokens
        self.total_cost_usd += call_cost

        record = {
            "call_id": len(self.call_history) + 1,
            "session_call_id": len(self.session_calls) + 1,
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "call_cost_usd": call_cost,
            "cumulative_cost_usd": self.total_cost_usd,
        }
        self.call_history.append(record)
        self.session_calls.append(record)

        log_msg = (
            f"[LLM Cost] #{record['call_id']} ({self.session_name}) | Model: '{model_name}' | "
            f"Prompt: {input_tokens:,} | Completion: {output_tokens:,} | "
            f"Cost: ${call_cost:.6f} | Session Total: ${self.total_cost_usd:.6f}"
        )
        logger.info(log_msg)
        print(f"\033[33m{log_msg}\033[0m")

        return record

    def end_session(
        self,
        session_name: Optional[str] = None,
        md_filepath: Optional[str] = None,
        force: bool = False,
    ) -> Optional[dict]:
        """Finalize the current session, write total token & cost usage to an md file, and reset session state."""
        if not self.session_calls and not force:
            return None

        name = session_name or self.session_name or "LLM General Session"
        start_time = self.session_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        session_prompt_tokens = sum(c["input_tokens"] for c in self.session_calls)
        session_completion_tokens = sum(c["output_tokens"] for c in self.session_calls)
        session_total_tokens = session_prompt_tokens + session_completion_tokens
        session_cost = sum(c["call_cost_usd"] for c in self.session_calls)
        num_calls = len(self.session_calls)

        # Unique models used
        models_used = sorted(list(set(c["model"] for c in self.session_calls))) if self.session_calls else ["N/A"]

        summary = {
            "session_name": name,
            "start_time": start_time,
            "end_time": end_time,
            "num_calls": num_calls,
            "models_used": ", ".join(models_used),
            "input_tokens": session_prompt_tokens,
            "output_tokens": session_completion_tokens,
            "total_tokens": session_total_tokens,
            "session_cost_usd": session_cost,
            "lifetime_cumulative_cost_usd": self.total_cost_usd,
        }

        # Write to markdown report
        target_file = md_filepath or DEFAULT_MD_FILENAME
        self._append_markdown_summary(target_file, summary)

        logger.info(
            "Finalized session '%s': %d API calls, %d tokens, $%.6f USD",
            name, num_calls, session_total_tokens, session_cost
        )
        print(f"\033[92m[LLM Session Completed] '{name}': {num_calls} calls, {session_total_tokens:,} tokens, ${session_cost:.6f} USD -> Logged to {target_file}\033[0m")

        # Reset session state
        self.session_name = None
        self.session_start_time = None
        self.session_calls = []

        return summary

    def _append_markdown_summary(self, filepath: str, summary: dict) -> None:
        """Append or update a clean markdown summary report table."""
        file_exists = os.path.exists(filepath)

        lines = []
        if not file_exists:
            lines.append("# 💰 GCSE AI — Token Usage & Cost Tracking Log")
            lines.append("")
            lines.append("> Tracks token consumption, LLM model providers, and estimated USD costs per operation session.")
            lines.append("")
            lines.append("## 📊 Session History")
            lines.append("")
            lines.append("| Session Name | Start Time | End Time | Calls | Models | Prompt Tokens | Completion Tokens | Total Tokens | Session Cost (USD) |")
            lines.append("| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |")

        row = (
            f"| **{summary['session_name']}** | `{summary['start_time']}` | `{summary['end_time']}` | "
            f"{summary['num_calls']} | `{summary['models_used']}` | {summary['input_tokens']:,} | "
            f"{summary['output_tokens']:,} | **{summary['total_tokens']:,}** | **${summary['session_cost_usd']:.6f}** |"
        )
        lines.append(row)

        try:
            with open(filepath, "a" if file_exists else "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            logger.error("Failed to write token cost summary to %s: %s", filepath, e)


# Global singleton instance
global_tracker = TokenCostTracker()


def _flush_session_at_exit():
    """Ensure any active in-flight session is written before Python process terminates."""
    if global_tracker.session_calls:
        global_tracker.end_session()


atexit.register(_flush_session_at_exit)
