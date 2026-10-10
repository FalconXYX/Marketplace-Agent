"""Tool implementations, in the order the model sees them."""
from marketplace_agent.tools.base import Tool
from marketplace_agent.tools.impl.deals import (
    AcceptBundleTool,
    AcceptOfferTool,
    BookPickupTool,
    CompleteSaleTool,
    ReleaseCommitmentTool,
)
from marketplace_agent.tools.impl.info import GetStatsTool, QueryCompsTool
from marketplace_agent.tools.impl.listing import PublishTool, RelistTool, RewriteListingTool
from marketplace_agent.tools.impl.messaging import SendMessageTool
from marketplace_agent.tools.impl.meta import EscalateTool, RememberTool, ScheduleWakeTool
from marketplace_agent.tools.impl.pricing import UpdatePriceTool

TOOL_CLASSES: tuple[type[Tool], ...] = (
    SendMessageTool,
    UpdatePriceTool,
    RewriteListingTool,
    PublishTool,
    RelistTool,
    AcceptOfferTool,
    AcceptBundleTool,
    BookPickupTool,
    ReleaseCommitmentTool,
    CompleteSaleTool,
    GetStatsTool,
    QueryCompsTool,
    EscalateTool,
    RememberTool,
    ScheduleWakeTool,
)

__all__ = [
    "TOOL_CLASSES",
    "AcceptBundleTool",
    "AcceptOfferTool",
    "BookPickupTool",
    "CompleteSaleTool",
    "EscalateTool",
    "GetStatsTool",
    "PublishTool",
    "QueryCompsTool",
    "ReleaseCommitmentTool",
    "RelistTool",
    "RememberTool",
    "RewriteListingTool",
    "ScheduleWakeTool",
    "SendMessageTool",
    "UpdatePriceTool",
]
