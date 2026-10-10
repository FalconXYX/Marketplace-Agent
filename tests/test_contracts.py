from sqlmodel import select

from marketplace_agent.agent.plan import Plan
from marketplace_agent.core.models import Item


def test_schema_creates_and_roundtrips(session, clock):
    session.add(Item(hint="desk", created_at=clock.now(), updated_at=clock.now()))
    session.commit()
    assert session.exec(select(Item)).one().hint == "desk"


def test_plan_from_wire_parses_args_json():
    plan = Plan.from_wire(
        {"rationale": "r", "actions": [{"tool": "update_price", "args_json": '{"price_cents": 10500}'},
                                        {"tool": "x", "args_json": "not json"}]}
    )
    assert plan.actions[0].args == {"price_cents": 10500}
    assert plan.actions[1].parse_error
