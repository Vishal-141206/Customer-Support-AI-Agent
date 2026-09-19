"""
Customer Support AI Agent
=========================
AWS Bedrock AgentCore customer support agent with:

- AgentCore Gateway for order tracking and refunds
- AgentCore Knowledge Base for product/policy information
- AgentCore Memory for customer context
- AgentCore Code Interpreter for loyalty calculations
- AgentCore Browser for web browsing
"""

# ── Imports ───────────────────────────────────────────────────────────────────

from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client

import argparse
import asyncio
import boto3
import json
import logging
import os
import uuid

from typing import Dict

from strands.hooks import (
    HookProvider,
    AfterInvocationEvent,
    HookRegistry,
    MessageAddedEvent,
)

from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")


# ── App Initialisation ────────────────────────────────────────────────────────

app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts.
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── Configuration ─────────────────────────────────────────────────────────────

GATEWAY_URL = (
    "https://customersupportgateway-ajtjpaciex"
    ".gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
)

KB_ID = "J0O9VYSPQJ"

REGION = "us-east-1"

MEMORY_ID = "CustomerSupportMemory-WmYIyYFjye"

# AWS-managed AgentCore Browser.
BROWSER_IDENTIFIER = "aws.browser.v1"


# ── Model and Clients ─────────────────────────────────────────────────────────

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(
    model_id=model_id
)

memory_client = MemoryClient(
    region_name=REGION
)

_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)


# ── Browser ───────────────────────────────────────────────────────────────────
#
# IMPORTANT:
# Initialize the AgentCore Browser once at module level instead of creating
# a new Browser object inside every AgentCore invocation.
#
# This follows the AWS Strands integration pattern and avoids repeatedly
# constructing the BrowserClient during each async request.
#

agent_core_browser = AgentCoreBrowser(
    region=REGION,
    identifier=BROWSER_IDENTIFIER,
)


# ── Namespace Helper ──────────────────────────────────────────────────────────

def get_namespaces(
    mem_client: MemoryClient,
    memory_id: str,
) -> Dict:
    """Return a dict mapping strategy type to namespace template."""

    strategies = mem_client.get_memory_strategies(memory_id)

    return {
        strategy["type"]: strategy["namespaces"][0]
        for strategy in strategies
    }


# ── Memory Hook ───────────────────────────────────────────────────────────────

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id

        self.namespaces = get_namespaces(
            memory_client,
            memory_id,
        )

    def retrieve_customer_context(
        self,
        event: MessageAddedEvent,
    ):
        """Retrieve relevant memories and prepend them to the user message."""

        messages = event.agent.messages

        if not messages:
            return

        last_message = messages[-1]

        if last_message.get("role") != "user":
            return

        content = last_message.get("content", [])

        if not content:
            return

        # Extract plain-text user message.
        query_parts = []

        for block in content:
            if isinstance(block, dict) and "text" in block:
                query_parts.append(block["text"])

        if not query_parts:
            return

        query = "\n".join(query_parts)

        memories = []

        for strategy_type, namespace_template in self.namespaces.items():

            namespace = namespace_template.replace(
                "{actorId}",
                self.actor_id,
            )

            try:
                results = self.memory_client.retrieve_memories(
                    self.memory_id,
                    namespace,
                    query,
                    top_k=10,
                )

                for result in results:

                    memory_text = (
                        result.get("content", {}).get("text")
                        if isinstance(result, dict)
                        else None
                    )

                    if memory_text:
                        memories.append(
                            f"[{strategy_type}] {memory_text}"
                        )

            except Exception as e:
                logger.warning(
                    "Memory retrieval failed for %s: %s",
                    strategy_type,
                    e,
                )

        if memories:

            context = "\n".join(memories)

            new_text = (
                "Customer Context:\n"
                f"{context}\n\n"
                f"{query}"
            )

            last_message["content"] = [
                {
                    "text": new_text
                }
            ]

    def save_support_interaction(
        self,
        event: AfterInvocationEvent,
    ):
        """Save the completed turn to memory after the agent responds."""

        messages = event.agent.messages

        customer_query = None
        assistant_response = None

        # Walk backwards through the conversation.
        for message in reversed(messages):

            role = message.get("role")
            content = message.get("content", [])

            text_parts = []

            if isinstance(content, list):

                for block in content:

                    if (
                        isinstance(block, dict)
                        and "text" in block
                    ):
                        text_parts.append(block["text"])

            text = "\n".join(text_parts).strip()

            if not text:
                continue

            if customer_query is None and role == "user":
                customer_query = text

            elif (
                assistant_response is None
                and role == "assistant"
            ):
                assistant_response = text

            if customer_query and assistant_response:
                break

        if not customer_query or not assistant_response:
            return

        try:

            self.memory_client.create_event(
                self.memory_id,
                self.actor_id,
                self.session_id,
                messages=[
                    (
                        customer_query,
                        "USER",
                    ),
                    (
                        assistant_response,
                        "ASSISTANT",
                    ),
                ],
            )

        except Exception as e:

            logger.warning(
                "Failed to save interaction to memory: %s",
                e,
            )

    def register_hooks(
        self,
        registry: HookRegistry,
    ) -> None:
        """Register memory callbacks."""

        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context,
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )


# ── Knowledge Base Tool ───────────────────────────────────────────────────────

@tool
def search_knowledge_base(
    query: str,
) -> str:
    """
    Search the Amazon product catalog and support knowledge base.

    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for.

    Returns:
        Relevant information retrieved from the knowledge base.
    """

    if not KB_ID:
        return "Knowledge base not configured."

    try:

        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={
                "text": query,
            },
        )

        results = response.get(
            "retrievalResults",
            [],
        )

        if not results:
            return (
                "No relevant information was found "
                "in the knowledge base."
            )

        chunks = []

        for result in results:

            content = result.get(
                "content",
                {},
            )

            if isinstance(content, dict):
                text = content.get(
                    "text",
                    "",
                )
            else:
                text = str(content)

            if text:
                chunks.append(text)

        if not chunks:
            return (
                "No relevant information was found "
                "in the knowledge base."
            )

        return "\n---\n".join(chunks)

    except Exception as e:

        logger.warning(
            "Knowledge base search failed: %s",
            e,
        )

        return (
            f"Knowledge base search failed: {e}"
        )


# ── Loyalty Discount Tool ────────────────────────────────────────────────────

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter.

    Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:
            Customer's current points balance.

        tier:
            Customer tier — Silver, Gold, or Platinum.

        order_total:
            Order total in USD.

        product_category:
            standard, device, or fresh.

    Returns:
        Full discount breakdown and final price.
    """

    code = f"""
import json
import math

loyalty_points = {int(loyalty_points)}
tier = {json.dumps(tier)}
order_total = {float(order_total)}
product_category = {json.dumps(product_category)}

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5
}}

tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15
}}

# 100 points = $1.
# Points can be redeemed in blocks of 500.
# Maximum redemption is 50% of the order value.

max_points_by_order = int(
    (order_total * 0.50) * 100
)

max_redeemable_points = min(
    loyalty_points,
    max_points_by_order
)

points_redeemed = (
    max_redeemable_points // 500
) * 500

points_discount = points_redeemed / 100

subtotal_after_points = (
    order_total - points_discount
)

tier_discount_rate = tier_rates.get(
    tier,
    0.00
)

tier_discount = (
    subtotal_after_points
    * tier_discount_rate
)

final_total = (
    subtotal_after_points
    - tier_discount
)

total_savings = (
    order_total
    - final_total
)

points_earned = math.floor(
    final_total
    * earn_rates.get(
        product_category,
        1
    )
)

remaining_points = (
    loyalty_points
    - points_redeemed
)

result = {{
    "loyalty_points": loyalty_points,
    "tier": tier,
    "order_total": round(order_total, 2),
    "product_category": product_category,
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}

print(json.dumps(result))
"""

    try:

        with code_session(REGION) as session:

            response = session.invoke(
                "executeCode",
                {
                    "language": "python",
                    "code": code,
                    "clearContext": True,
                },
            )

            # AgentCore Code Interpreter returns execution
            # events under the "stream" key.

            for event in response.get(
                "stream",
                [],
            ):

                if "result" in event:

                    return json.dumps(
                        event["result"],
                        default=str,
                    )

            return (
                "Code Interpreter returned no result."
            )

    except Exception as e:

        logger.warning(
            "Code Interpreter unavailable: %s",
            e,
        )

        # Required fallback.
        tier_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15,
        }

        tier_discount_rate = tier_rates.get(
            tier,
            0.00,
        )

        tier_discount = (
            order_total
            * tier_discount_rate
        )

        final_total = (
            order_total
            - tier_discount
        )

        fallback = {
            "order_total": round(
                order_total,
                2,
            ),
            "tier": tier,
            "tier_discount": round(
                tier_discount,
                2,
            ),
            "tier_discount_pct": round(tier_discount_rate * 100, 2),
            "final_total": round(
                final_total,
                2,
            ),
            "total_savings": round(
                tier_discount,
                2,
            ),
            "note": (
                "Code Interpreter unavailable; "
                "tier discount only."
            ),
        }

        return json.dumps(fallback)


# ── Agent Entrypoint ──────────────────────────────────────────────────────────

@app.entrypoint
async def invoke(
    payload,
    context=None,
):
    """
    Main handler called by AgentCore.

    Expected payload keys:

        prompt:
            Customer message.

        customer_id:
            Unique customer identifier.

        session_id:
            Session identifier.
            Generated automatically if missing.
    """

    try:

        user_input = payload.get(
            "prompt",
            "",
        )

        actor_id = payload.get(
            "customer_id",
            "anonymous",
        )

        session_id = payload.get(
            "session_id",
            str(uuid.uuid4()),
        )

        if not user_input:
            return "Please provide a prompt."

        # ── Memory ───────────────────────────────────────────────────────────

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # ── Tools ────────────────────────────────────────────────────────────
        #
        # IMPORTANT:
        # Use the single module-level AgentCoreBrowser instance.
        #

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        # ── AgentCore Gateway ────────────────────────────────────────────────

        gateway_client = MCPClient(
            lambda: streamable_http_client(
                GATEWAY_URL
            )
        )

        with gateway_client:

            gateway_tools = (
                gateway_client.list_tools_sync()
            )

            tools.extend(
                gateway_tools
            )

            # ── Strands Agent ────────────────────────────────────────────────

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=(
                    "You are a helpful customer support AI agent. "

                    "Assist customers with order tracking, refunds, "
                    "product information, loyalty discounts, web "
                    "browsing, and general support questions. "

                    "Use the available tools when appropriate. "

                    "For order information, use the Gateway tools. "

                    "For product specifications, policies, warranty "
                    "information, and loyalty program information, use "
                    "the knowledge base. "

                    "For loyalty discount calculations, ALWAYS use "
                    "the calculate_loyalty_discount tool and report "
                    "its result rather than calculating the discount "
                    "yourself. "

                    "For requests to visit or inspect a website, use "
                    "the AgentCore browser tool. "

                    "When using the browser, navigate to the URL "
                    "provided by the customer and retrieve the "
                    "requested information. "

                    "Be accurate, concise, and helpful."
                ),
            )

            # ── Invoke Agent ─────────────────────────────────────────────────

            response = await agent.invoke_async(
                user_input
            )

        # ── Return response text ─────────────────────────────────────────────

        if hasattr(
            response,
            "content",
        ):

            content = response.content

            if (
                isinstance(content, list)
                and content
            ):

                first_block = content[0]

                if isinstance(
                    first_block,
                    dict,
                ):

                    return first_block.get(
                        "text",
                        str(first_block),
                    )

                return str(first_block)

            return str(content)

        return str(response)

    except Exception as e:

        logger.exception(
            "Agent invocation failed"
        )

        return (
            f"Agent invocation failed: {e}"
        )


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    """Run one invocation from the command line."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "payload",
        type=str,
    )

    args = parser.parse_args()

    response = asyncio.run(
        invoke(
            json.loads(
                args.payload
            )
        )
    )

    print(response)


# AgentCore deployment entry point.
if __name__ == "__main__":
    app.run()