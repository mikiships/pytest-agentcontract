"""A minimal customer support agent for demonstrating pytest-agentcontract.

This is NOT a real agent -- it's a simple simulation that makes deterministic
tool calls to demonstrate the record/replay/assert workflow.
"""

from __future__ import annotations

from typing import Any


# Simulated tools (in a real app these would hit databases/APIs)
ORDERS_DB = {
    "ORD-123": {
        "id": "ORD-123",
        "customer": "alice@example.com",
        "items": [{"name": "Wireless Headphones", "price": 79.99}],
        "total": 79.99,
        "status": "delivered",
        "delivered_at": "2026-02-10",
    },
    "ORD-456": {
        "id": "ORD-456",
        "customer": "bob@example.com",
        "items": [{"name": "USB-C Cable", "price": 12.99}],
        "total": 12.99,
        "status": "shipped",
        "delivered_at": None,
    },
}

REFUND_POLICY = {
    "window_days": 30,
    "eligible_statuses": ["delivered"],
}


def lookup_order(order_id: str) -> dict[str, Any]:
    """Look up an order by ID."""
    order = ORDERS_DB.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found"}
    return order


def check_refund_eligibility(order_id: str) -> dict[str, Any]:
    """Check if an order is eligible for refund."""
    order = ORDERS_DB.get(order_id)
    if order is None:
        return {"eligible": False, "reason": "Order not found"}
    if order["status"] not in REFUND_POLICY["eligible_statuses"]:
        return {"eligible": False, "reason": f"Order status is '{order['status']}', not delivered"}
    return {"eligible": True, "order_id": order_id, "amount": order["total"]}


def process_refund(order_id: str, amount: float, method: str = "original") -> dict[str, Any]:
    """Process a refund for an order."""
    return {
        "success": True,
        "refund_id": f"REF-{order_id}",
        "amount": amount,
        "method": method,
    }


# The "agent" -- a simple state machine that calls tools
def run_support_agent(user_message: str, tools: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run a simple support agent that handles refund requests.

    Returns a list of turns (messages + tool calls) for recording.
    """
    if tools is None:
        tools = {
            "lookup_order": lookup_order,
            "check_refund_eligibility": check_refund_eligibility,
            "process_refund": process_refund,
        }

    turns: list[dict[str, Any]] = []

    # Turn 0: User message
    turns.append({"role": "user", "content": user_message})

    # Extract order ID (simple pattern matching, not real NLU)
    order_id = None
    for word in user_message.split():
        if word.startswith("ORD-"):
            order_id = word.rstrip(".,!?")
            break

    if order_id is None:
        turns.append({
            "role": "assistant",
            "content": "I'd be happy to help with a refund. Could you provide your order ID?",
        })
        return turns

    # Turn 1: Look up the order
    order = tools["lookup_order"](order_id)
    turns.append({
        "role": "assistant",
        "content": f"Let me look up order {order_id}.",
        "tool_calls": [{
            "id": "tc_lookup",
            "function": "lookup_order",
            "arguments": {"order_id": order_id},
            "result": order,
        }],
    })

    if "error" in order:
        turns.append({
            "role": "assistant",
            "content": f"I'm sorry, I couldn't find order {order_id}.",
        })
        return turns

    # Turn 2: Check eligibility
    eligibility = tools["check_refund_eligibility"](order_id)
    turns.append({
        "role": "assistant",
        "content": "Checking refund eligibility...",
        "tool_calls": [{
            "id": "tc_eligibility",
            "function": "check_refund_eligibility",
            "arguments": {"order_id": order_id},
            "result": eligibility,
        }],
    })

    if not eligibility.get("eligible"):
        reason = eligibility.get("reason", "Unknown reason")
        turns.append({
            "role": "assistant",
            "content": f"I'm sorry, this order isn't eligible for a refund. Reason: {reason}",
        })
        return turns

    # Turn 3: User confirms
    turns.append({"role": "user", "content": "Yes, please process the refund."})

    # Turn 4: Process refund
    amount = eligibility["amount"]
    refund = tools["process_refund"](order_id, amount)
    turns.append({
        "role": "assistant",
        "content": f"Your refund of ${amount:.2f} has been processed. Refund ID: {refund['refund_id']}",
        "tool_calls": [{
            "id": "tc_refund",
            "function": "process_refund",
            "arguments": {"order_id": order_id, "amount": amount, "method": "original"},
            "result": refund,
        }],
    })

    return turns
