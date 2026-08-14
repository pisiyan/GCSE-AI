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
        """Start a new session for grouping LLM calls under a named task (e.g. 'Chatbot', 'Exam Generation')."""
        if self.session_calls:
            # End previous session before starting a new one
            self.end_session()

        self.session_name = session_name or "LLM Session"
        self.session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_calls = []

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
            self.start_session("LLM Session")

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
            f"[LLM API Call #{record['call_id']}] Model: '{model_name}' | "
            f"Input Tokens: {input_tokens:,} | Output Tokens: {output_tokens:,} | "
            f"Call Cost: ${call_cost:.6f} | Session Total: ${self.total_cost_usd:.6f}"
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

        name = session_name or self.session_name or "LLM Session"
        timestamp = self.session_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        total_input = sum(r["input_tokens"] for r in self.session_calls)
        total_output = sum(r["output_tokens"] for r in self.session_calls)
        total_tokens = total_input + total_output
        total_cost = sum(r["call_cost_usd"] for r in self.session_calls)
        models_used = sorted(list(set(r["model"] for r in self.session_calls if "model" in r)))
        models_str = ", ".join(models_used) if models_used else "N/A"

        session_summary = {
            "timestamp": timestamp,
            "session_name": name,
            "models": models_str,
            "calls": len(self.session_calls),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_tokens": total_tokens,
            "cost_usd": total_cost,
        }

        saved_file = self._save_session_to_md(session_summary, md_filepath=md_filepath)

        msg = (
            f"\033[32m[TokenCostTracker] Finished Session '{name}': "
            f"{len(self.session_calls)} call(s) | {total_tokens:,} tokens | ${total_cost:.6f} USD -> Updated {saved_file}\033[0m"
        )
        print(msg)
        logger.info(msg)

        # Reset session state
        self.session_name = None
        self.session_start_time = None
        self.session_calls = []

        return session_summary

    def _save_session_to_md(self, session_summary: dict, md_filepath: Optional[str] = None) -> str:
        """Append session summary to a markdown file and update grand total section."""
        target_path = md_filepath or os.environ.get("TOKEN_COST_MD_FILE", DEFAULT_MD_FILENAME)
        if not os.path.isabs(target_path):
            project_dir = os.path.dirname(os.path.abspath(__file__))
            target_path = os.path.join(project_dir, target_path)

        existing_rows: list[dict] = []

        if os.path.exists(target_path):
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Parse existing markdown table rows if present
            row_pattern = r"^\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|\s*\$([\d\.]+)\s*\|"
            for line in content.splitlines():
                match = re.match(row_pattern, line.strip())
                if match:
                    existing_rows.append({
                        "timestamp": match.group(1),
                        "session_name": match.group(2),
                        "models": match.group(3),
                        "calls": int(match.group(4)),
                        "input_tokens": int(match.group(5).replace(",", "")),
                        "output_tokens": int(match.group(6).replace(",", "")),
                        "total_tokens": int(match.group(7).replace(",", "")),
                        "cost_usd": float(match.group(8)),
                    })

        # Append current session summary to existing rows
        existing_rows.append(session_summary)

        # Calculate Grand Totals
        total_sessions = len(existing_rows)
        total_calls = sum(r["calls"] for r in existing_rows)
        total_input = sum(r["input_tokens"] for r in existing_rows)
        total_output = sum(r["output_tokens"] for r in existing_rows)
        total_tokens = sum(r["total_tokens"] for r in existing_rows)
        total_cost = sum(r["cost_usd"] for r in existing_rows)

        # Generate fresh Markdown content
        lines = [
            "# LLM Token & Cost Usage - Session Log",
            "",
            "| Date & Time | Session Name | Models Used | Calls | Input Tokens | Output Tokens | Total Tokens | Cost (USD) |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for row in existing_rows:
            lines.append(
                f"| {row['timestamp']} | {row['session_name']} | {row['models']} | "
                f"{row['calls']} | {row['input_tokens']:,} | {row['output_tokens']:,} | "
                f"{row['total_tokens']:,} | ${row['cost_usd']:.6f} |"
            )

        lines.extend([
            "",
            "## Cumulative Summary Across All Sessions",
            f"- **Total Sessions Logged**: {total_sessions}",
            f"- **Total API Calls**: {total_calls}",
            f"- **Total Input Tokens**: {total_input:,}",
            f"- **Total Output Tokens**: {total_output:,}",
            f"- **Total Combined Tokens**: {total_tokens:,}",
            f"- **Total Cumulative Cost (USD)**: ${total_cost:.6f}",
            "",
        ])

        with open(target_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return target_path

    def get_summary(self) -> dict:
        """Get cumulative usage and cost summary dictionary."""
        return {
            "total_calls": len(self.call_history),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
        }

    def print_summary_report(self) -> None:
        """Print a structured report of all LLM calls and costs."""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("           LLM API TOKEN USAGE & COST REPORT           ")
        print("=" * 60)
        print(f" Total API Calls Made:        {summary['total_calls']}")
        print(f" Total Input (Prompt) Tokens: {summary['total_prompt_tokens']:,}")
        print(f" Total Output Tokens:        {summary['total_completion_tokens']:,}")
        print(f" Combined Tokens:            {summary['total_tokens']:,}")
        print(f" Total Cost (USD):           ${summary['total_cost_usd']:.6f}")
        print("=" * 60 + "\n")


# Global tracker instance
global_tracker = TokenCostTracker()


def _cleanup_session_on_exit() -> None:
    """Atexit handler to ensure active sessions are logged on process termination."""
    if global_tracker and global_tracker.session_calls:
        global_tracker.end_session()


atexit.register(_cleanup_session_on_exit)


if __name__ == "__main__":
    print("Demonstrating TokenCostTracker with Session Logging:")
    tracker = TokenCostTracker()
    tracker.start_session("Exam Generation")
    tracker.track_call("gpt-4o-mini", "Generate 5 biology questions", "Question 1...", prompt_tokens=150, completion_tokens=45)
    tracker.track_call("gpt-4o", "Generate mark scheme", "Mark scheme...", prompt_tokens=600, completion_tokens=350)
    tracker.end_session()
    tracker.print_summary_report()

