"""Tool registry and the action agent: per-tool Pydantic argument schemas, risk and approval metadata, dry-run executors, and the code that turns an intent into a ProposedAction. Proposes, never executes.

The inversion worth stating out loud: the dangerous agentic pattern is a broad
tool surface with an LLM deciding permissions. Here the LLM proposes, a
deterministic policy layer authorises, and the surface is four tools wide and
mostly reversible.

Every executor is a dry run. There is no code path in this repository that moves
money, changes a credential, or alters a credit term.
"""

from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from src.schemas import ProposedAction, RiskTier


class OrderReplacementCardArgs(BaseModel):
    reason: str = Field(description="Why a replacement is needed, e.g. 'damaged'.")
    delivery_address_on_file: bool = Field(
        default=True, description="False would require an address change first, which is a separate tool."
    )


class UpdateContactDetailsArgs(BaseModel):
    field: str = Field(description="Which detail to change: 'phone' or 'email'.")
    new_value_redacted: str = Field(description="The new value, already PII-redacted by ingress.")


class RequestCardLimitChangeArgs(BaseModel):
    new_limit_dkk: int = Field(ge=0, description="Requested limit in DKK.")


class BlockCardArgs(BaseModel):
    card_last_four: str = Field(description="Last four digits, the only card identifier we accept.")
    reason: str = Field(description="Why the card is being blocked.")


def dry_run(tool_name: str, arguments: BaseModel) -> str:
    """Return a simulated receipt. This is the only executor in the registry.

    It deliberately takes the same shape a real executor would, so the seam where
    a real one would be substituted is visible — and equally visible that it is not.
    """
    return f"DRY RUN: {tool_name} would have been called with {arguments.model_dump()}. Nothing was executed."


class Tool(BaseModel):
    """One registry entry. Four fields: what it is, what it takes, who must approve, what runs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Registry key, used as ProposedAction.tool_name.")
    args_schema: type[BaseModel] = Field(description="Pydantic schema its arguments must satisfy.")
    requires_human_approval: bool = Field(
        description="Decided here in code, never by the model. True for anything "
        "irreversible, money-moving, or credential-changing."
    )
    executor: Callable[[str, BaseModel], str] = Field(description="Dry-run only.")


REGISTRY: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        # Reversible, low blast radius: a spare card in the post is undoable.
        Tool(
            name="order_replacement_card",
            args_schema=OrderReplacementCardArgs,
            requires_human_approval=False,
            executor=dry_run,
        ),
        # Reversible, but a contact-detail change is a known account-takeover step,
        # so it is proposed rather than auto-confirmed.
        Tool(
            name="update_contact_details",
            args_schema=UpdateContactDetailsArgs,
            requires_human_approval=True,
            executor=dry_run,
        ),
        # Alters a credit term. Assumption 6 forbids this outright; the tool exists
        # to show the approval gate, and can never be auto-confirmed.
        Tool(
            name="request_card_limit_change",
            args_schema=RequestCardLimitChangeArgs,
            requires_human_approval=True,
            executor=dry_run,
        ),
        # Credential-adjacent and hard to undo for the customer mid-transaction.
        Tool(
            name="block_card",
            args_schema=BlockCardArgs,
            requires_human_approval=True,
            executor=dry_run,
        ),
    )
}

# Which tool, if any, an intent may propose. A mapping in code rather than a
# choice the model makes: an intent that is not listed here has no tool at all.
INTENT_TO_TOOL: dict[str, str] = {
    "order_physical_card": "order_replacement_card",
    "get_physical_card": "order_replacement_card",
    "getting_spare_card": "order_replacement_card",
    "get_disposable_virtual_card": "order_replacement_card",
    "getting_virtual_card": "order_replacement_card",
    "edit_personal_details": "update_contact_details",
}


def propose_action(intent: str) -> ProposedAction | None:
    """Build a ProposedAction for an intent, or None when no tool applies.

    Arguments are left empty on purpose. Filling them needs details the customer
    has not been asked for yet, and inventing them is exactly the failure this
    whole design is arranged to prevent.
    """
    tool_name = INTENT_TO_TOOL.get(intent)
    if tool_name is None:
        return None

    tool = REGISTRY[tool_name]
    return ProposedAction(
        tool_name=tool.name,
        arguments={},
        risk=RiskTier.MEDIUM if tool.requires_human_approval else RiskTier.LOW,
        reversible=not tool.requires_human_approval,
        requires_auth_level="strong" if tool.requires_human_approval else "basic",
        requires_human_approval=tool.requires_human_approval,
        dry_run_receipt=tool.executor(tool.name, tool.args_schema.model_construct()),
    )
