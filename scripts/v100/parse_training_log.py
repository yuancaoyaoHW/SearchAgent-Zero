#!/usr/bin/env python3
"""
SearchAgent-Zero: Training Log Parser

Parses training logs and outputs JSON for the training-tracker frontend.
Also provides real-time monitoring and anomaly detection.

Usage:
    # Parse a log file and output JSON
    python parse_training_log.py --log logs/experiment.log --output metrics.json

    # Watch a log file in real-time (prints alerts)
    python parse_training_log.py --log logs/experiment.log --watch

    # Generate a summary report
    python parse_training_log.py --log logs/experiment.log --report
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ============================================================
# Data Structures
# ============================================================

@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int = 0
    timestamp: str = ""
    time_seconds: float = 0.0

    # Core metrics
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    actor_loss: Optional[float] = None
    actor_entropy: Optional[float] = None
    kl_divergence: Optional[float] = None
    grad_norm: Optional[float] = None

    # Tool call metrics
    tool_call_success_rate: Optional[float] = None
    tool_call_turn_mean: Optional[float] = None
    all_call_tool_counts: Optional[float] = None
    tool_call_success_counts: Optional[float] = None

    # Abnormal trajectory metrics
    tool_parser_error_pct: Optional[float] = None
    searched_query_pct: Optional[float] = None
    too_many_tool_call_pct: Optional[float] = None
    duplicate_search_result_pct: Optional[float] = None
    too_many_turn_pct: Optional[float] = None
    too_long_seq_truncated_pct: Optional[float] = None
    response_truncated_pct: Optional[float] = None

    # Throughput
    tokens_per_second: Optional[float] = None
    rollout_time: Optional[float] = None
    actor_update_time: Optional[float] = None
    ref_logprob_time: Optional[float] = None


@dataclass
class TrainingSession:
    """A complete training session."""
    experiment_name: str = ""
    recipe: str = ""  # "search_r1" or "asearch"
    model: str = ""
    start_time: str = ""
    steps: list = field(default_factory=list)
    alerts: list = field(default_factory=list)


# ============================================================
# Log Parsing
# ============================================================

# Regex patterns for metric extraction
PATTERNS = {
    # Step header patterns (adapt to actual log format)
    "step": [
        re.compile(r"[Ss]tep\s+(\d+)"),
        re.compile(r"global_step[=:\s]+(\d+)"),
        re.compile(r"\[step\s+(\d+)\]"),
    ],
    # Metric patterns: key: value or key=value
    "metric": re.compile(r"([\w/]+)[=:\s]+([+-]?\d+\.?\d*(?:e[+-]?\d+)?|nan|inf|-inf)"),
    # Timestamp
    "timestamp": re.compile(r"\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]?"),
    # Time per step
    "step_time": [
        re.compile(r"[Tt]ime[=:\s]+(\d+)m(\d+)s"),
        re.compile(r"step_time[=:\s]+(\d+\.?\d*)"),
        re.compile(r"elapsed[=:\s]+(\d+\.?\d*)s"),
    ],
}

# Metric name mapping (log key → StepMetrics field)
METRIC_MAP = {
    "reward/mean": "reward_mean",
    "reward_mean": "reward_mean",
    "reward/std": "reward_std",
    "reward_std": "reward_std",
    "actor/loss": "actor_loss",
    "actor_loss": "actor_loss",
    "policy_loss": "actor_loss",
    "actor/entropy": "actor_entropy",
    "entropy": "actor_entropy",
    "kl_divergence": "kl_divergence",
    "kl": "kl_divergence",
    "approx_kl": "kl_divergence",
    "grad_norm": "grad_norm",
    "actor/grad_norm": "grad_norm",
    "turn/tool_call_success_rate/mean": "tool_call_success_rate",
    "tool_call_success_rate": "tool_call_success_rate",
    "turn/tool_call_turn/mean": "tool_call_turn_mean",
    "tool_call_turn": "tool_call_turn_mean",
    "turn/all_call_tool_counts/mean": "all_call_tool_counts",
    "turn/tool_call_success_counts/mean": "tool_call_success_counts",
    "abnormal_trajectory/tool_parser_error_count_percentage": "tool_parser_error_pct",
    "abnormal_trajectory/searched_query_count_percentage": "searched_query_pct",
    "abnormal_trajectory/too_many_tool_call_count_percentage": "too_many_tool_call_pct",
    "abnormal_trajectory/duplicate_search_result_count_percentage": "duplicate_search_result_pct",
    "abnormal_trajectory/too_many_turn_count_percentage": "too_many_turn_pct",
    "abnormal_trajectory/too_long_seq_truncated_count_percentage": "too_long_seq_truncated_pct",
    "abnormal_trajectory/response_truncated_count_percentage": "response_truncated_pct",
    "throughput/tokens_per_second": "tokens_per_second",
    "tokens_per_second": "tokens_per_second",
    "timing/rollout_s": "rollout_time",
    "timing/actor_update_s": "actor_update_time",
    "timing/ref_logprob_s": "ref_logprob_time",
}


def parse_step_time(line: str) -> Optional[float]:
    """Extract step time in seconds from a log line."""
    for pattern in PATTERNS["step_time"]:
        m = pattern.search(line)
        if m:
            groups = m.groups()
            if len(groups) == 2:  # XmYs format
                return int(groups[0]) * 60 + int(groups[1])
            else:  # seconds format
                return float(groups[0])
    return None


def parse_log_file(log_path: str) -> TrainingSession:
    """Parse a training log file into structured metrics."""
    session = TrainingSession()
    current_step = StepMetrics()
    current_step_num = -1

    # Try to infer experiment name from path
    log_name = Path(log_path).stem
    session.experiment_name = log_name
    if "search_r1" in log_name.lower() or "3b" in log_name.lower():
        session.recipe = "search_r1"
        session.model = "Qwen2.5-3B-Instruct"
    elif "asearch" in log_name.lower() or "8b" in log_name.lower():
        session.recipe = "asearch"
        session.model = "Qwen3-8B"

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check for step number
            step_num = None
            for pattern in PATTERNS["step"]:
                m = pattern.search(line)
                if m:
                    step_num = int(m.group(1))
                    break

            # If new step detected, save previous and start new
            if step_num is not None and step_num != current_step_num:
                if current_step_num >= 0:
                    session.steps.append(asdict(current_step))
                current_step = StepMetrics(step=step_num)
                current_step_num = step_num

            # Extract timestamp
            ts_match = PATTERNS["timestamp"].search(line)
            if ts_match and not current_step.timestamp:
                current_step.timestamp = ts_match.group(1)
                if not session.start_time:
                    session.start_time = ts_match.group(1)

            # Extract step time
            step_time = parse_step_time(line)
            if step_time is not None:
                current_step.time_seconds = step_time

            # Extract metrics
            for match in PATTERNS["metric"].finditer(line):
                key, value_str = match.group(1), match.group(2)
                if key in METRIC_MAP:
                    field_name = METRIC_MAP[key]
                    try:
                        if value_str in ("nan", "inf", "-inf"):
                            value = float(value_str)
                        else:
                            value = float(value_str)
                        setattr(current_step, field_name, value)
                    except (ValueError, TypeError):
                        pass

    # Don't forget the last step
    if current_step_num >= 0:
        session.steps.append(asdict(current_step))

    return session


# ============================================================
# Anomaly Detection
# ============================================================

@dataclass
class Alert:
    step: int
    severity: str  # "warning", "error", "critical"
    category: str
    message: str
    suggestion: str


def detect_anomalies(session: TrainingSession) -> list[dict]:
    """Analyze metrics and detect training anomalies."""
    alerts = []

    for i, step_data in enumerate(session.steps):
        step_num = step_data["step"]

        # --- NaN / Inf detection ---
        if step_data.get("actor_loss") is not None:
            import math
            if math.isnan(step_data["actor_loss"]) or math.isinf(step_data["actor_loss"]):
                alerts.append(asdict(Alert(
                    step=step_num, severity="critical", category="nan_loss",
                    message=f"Loss is NaN/Inf at step {step_num}",
                    suggestion="降低学习率到 5e-7，从上一个 checkpoint 恢复"
                )))

        # --- Gradient explosion ---
        if step_data.get("grad_norm") is not None and step_data["grad_norm"] > 50:
            alerts.append(asdict(Alert(
                step=step_num, severity="error", category="grad_explosion",
                message=f"Gradient norm = {step_data['grad_norm']:.1f} at step {step_num}",
                suggestion="添加 gradient clipping (max_grad_norm=1.0) 或降低学习率"
            )))

        # --- KL divergence too high ---
        if step_data.get("kl_divergence") is not None and step_data["kl_divergence"] > 3.0:
            alerts.append(asdict(Alert(
                step=step_num, severity="error", category="kl_explosion",
                message=f"KL divergence = {step_data['kl_divergence']:.3f} at step {step_num}",
                suggestion="增大 kl_loss_coef (0.001→0.01) 或降低学习率"
            )))

        # --- Reward stagnation ---
        if i >= 100 and step_data.get("reward_mean") is not None:
            # Check if reward hasn't improved in last 50 steps
            recent_rewards = [
                s.get("reward_mean") for s in session.steps[max(0, i-50):i]
                if s.get("reward_mean") is not None
            ]
            if recent_rewards and step_data["reward_mean"] is not None:
                avg_recent = sum(r for r in recent_rewards if r is not None) / max(len(recent_rewards), 1)
                if abs(step_data["reward_mean"] - avg_recent) < 0.01 and step_data["reward_mean"] < 0.2:
                    alerts.append(asdict(Alert(
                        step=step_num, severity="warning", category="reward_stagnation",
                        message=f"Reward stagnant at {step_data['reward_mean']:.3f} for 50+ steps",
                        suggestion="尝试增大学习率 (×2) 或增大 rollout.n"
                    )))

        # --- Entropy collapse ---
        if step_data.get("actor_entropy") is not None and step_data["actor_entropy"] < 0.5:
            alerts.append(asdict(Alert(
                step=step_num, severity="error", category="entropy_collapse",
                message=f"Entropy = {step_data['actor_entropy']:.3f} at step {step_num} (mode collapse risk)",
                suggestion="增大 kl_loss_coef，降低学习率，从早期 checkpoint 恢复"
            )))

        # --- Tool call failure ---
        if step_num > 50 and step_data.get("tool_call_success_rate") is not None:
            if step_data["tool_call_success_rate"] < 0.1:
                alerts.append(asdict(Alert(
                    step=step_num, severity="error", category="tool_failure",
                    message=f"Tool call success rate = {step_data['tool_call_success_rate']:.1%} at step {step_num}",
                    suggestion="检查检索服务是否正常，确认 format=hermes"
                )))

        # --- Step time anomaly ---
        if step_data.get("time_seconds") and step_data["time_seconds"] > 600:
            alerts.append(asdict(Alert(
                step=step_num, severity="warning", category="slow_step",
                message=f"Step took {step_data['time_seconds']:.0f}s (> 10 min) at step {step_num}",
                suggestion="检查检索服务延迟，或是否有 GPU 空闲"
            )))

    return alerts


# ============================================================
# Report Generation
# ============================================================

def generate_report(session: TrainingSession) -> str:
    """Generate a human-readable training report."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  训练报告: {session.experiment_name}")
    lines.append("=" * 60)
    lines.append(f"  配方: {session.recipe} | 模型: {session.model}")
    lines.append(f"  开始时间: {session.start_time}")
    lines.append(f"  总步数: {len(session.steps)}")
    lines.append("")

    if not session.steps:
        lines.append("  ⚠️ 未找到任何训练步骤数据")
        return "\n".join(lines)

    # Summary statistics
    rewards = [s["reward_mean"] for s in session.steps if s.get("reward_mean") is not None]
    tool_rates = [s["tool_call_success_rate"] for s in session.steps if s.get("tool_call_success_rate") is not None]
    step_times = [s["time_seconds"] for s in session.steps if s.get("time_seconds") and s["time_seconds"] > 0]

    lines.append("  📊 关键指标摘要")
    lines.append("  " + "-" * 40)
    if rewards:
        lines.append(f"  Reward:  初始 {rewards[0]:.3f} → 最终 {rewards[-1]:.3f} (最高 {max(rewards):.3f})")
    if tool_rates:
        lines.append(f"  Tool成功率: 初始 {tool_rates[0]:.1%} → 最终 {tool_rates[-1]:.1%}")
    if step_times:
        lines.append(f"  每步时间: 平均 {sum(step_times)/len(step_times):.0f}s (最快 {min(step_times):.0f}s, 最慢 {max(step_times):.0f}s)")
        total_hours = sum(step_times) / 3600
        lines.append(f"  总训练时间: {total_hours:.1f} 小时")
    lines.append("")

    # Alerts
    alerts = detect_anomalies(session)
    if alerts:
        lines.append(f"  ⚠️ 检测到 {len(alerts)} 个异常")
        lines.append("  " + "-" * 40)
        for alert in alerts[:10]:  # Show top 10
            icon = {"critical": "🔴", "error": "🟡", "warning": "⚪"}.get(alert["severity"], "⚪")
            lines.append(f"  {icon} Step {alert['step']}: {alert['message']}")
            lines.append(f"     → {alert['suggestion']}")
        if len(alerts) > 10:
            lines.append(f"  ... 还有 {len(alerts) - 10} 个异常")
    else:
        lines.append("  ✅ 未检测到异常，训练看起来正常！")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================
# Frontend JSON Export
# ============================================================

def export_for_frontend(session: TrainingSession) -> dict:
    """Export data in format compatible with training-tracker frontend."""
    alerts = detect_anomalies(session)
    session.alerts = alerts

    # Convert to frontend log format
    frontend_logs = []
    for step_data in session.steps:
        frontend_logs.append({
            "time": step_data.get("timestamp", ""),
            "name": session.experiment_name,
            "recipe": session.recipe,
            "step": step_data["step"],
            "reward": step_data.get("reward_mean"),
            "toolSuccess": step_data.get("tool_call_success_rate"),
            "notes": "",
        })

    return {
        "session": asdict(session),
        "frontend_logs": frontend_logs,
        "alerts": alerts,
        "summary": {
            "total_steps": len(session.steps),
            "final_reward": session.steps[-1].get("reward_mean") if session.steps else None,
            "final_tool_rate": session.steps[-1].get("tool_call_success_rate") if session.steps else None,
            "alert_count": len(alerts),
        }
    }


# ============================================================
# Watch Mode (Real-time Monitoring)
# ============================================================

def watch_log(log_path: str):
    """Watch a log file in real-time and print alerts."""
    print(f"👁️  Watching: {log_path}")
    print(f"   Press Ctrl+C to stop\n")

    last_pos = 0
    last_step = -1
    seen_alerts = set()

    while True:
        try:
            if not os.path.exists(log_path):
                time.sleep(2)
                continue

            with open(log_path, "r", errors="ignore") as f:
                f.seek(last_pos)
                new_content = f.read()
                last_pos = f.tell()

            if new_content:
                # Re-parse the full file for context
                session = parse_log_file(log_path)
                if session.steps:
                    latest = session.steps[-1]
                    step_num = latest["step"]

                    if step_num > last_step:
                        last_step = step_num
                        # Print step summary
                        reward = latest.get("reward_mean")
                        tool_rate = latest.get("tool_call_success_rate")
                        kl = latest.get("kl_divergence")
                        print(f"  Step {step_num:>4d} | "
                              f"reward: {reward:.3f if reward else '?':>7} | "
                              f"tool: {f'{tool_rate:.1%}' if tool_rate else '?':>6} | "
                              f"KL: {f'{kl:.3f}' if kl else '?':>6}")

                    # Check for new alerts
                    alerts = detect_anomalies(session)
                    for alert in alerts:
                        alert_key = f"{alert['step']}_{alert['category']}"
                        if alert_key not in seen_alerts:
                            seen_alerts.add(alert_key)
                            icon = {"critical": "🔴", "error": "🟡", "warning": "⚪"}[alert["severity"]]
                            print(f"\n  {icon} ALERT: {alert['message']}")
                            print(f"     → {alert['suggestion']}\n")

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n\n👋 Stopped watching.")
            break


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SearchAgent-Zero Training Log Parser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse log and export JSON for frontend
  python parse_training_log.py --log logs/experiment.log --output metrics.json

  # Watch log in real-time
  python parse_training_log.py --log logs/experiment.log --watch

  # Print summary report
  python parse_training_log.py --log logs/experiment.log --report

  # Import JSON into frontend (copy to browser console):
  #   1. Open training-tracker/index.html
  #   2. Click "导入数据"
  #   3. Select the metrics.json file
        """
    )
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch log in real-time")
    parser.add_argument("--report", "-r", action="store_true", help="Print summary report")

    args = parser.parse_args()

    if not os.path.exists(args.log) and not args.watch:
        print(f"Error: Log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    if args.watch:
        watch_log(args.log)
        return

    # Parse the log
    session = parse_log_file(args.log)

    if args.report:
        print(generate_report(session))
        return

    # Export JSON
    data = export_for_frontend(session)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Exported {len(session.steps)} steps to {args.output}")
        print(f"   Alerts: {len(data['alerts'])}")
        if data["summary"]["final_reward"] is not None:
            print(f"   Final reward: {data['summary']['final_reward']:.3f}")
    else:
        # Print to stdout
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        print()


if __name__ == "__main__":
    main()
