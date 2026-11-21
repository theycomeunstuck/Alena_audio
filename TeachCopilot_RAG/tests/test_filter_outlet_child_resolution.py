"""Audit F-058: Filter.outlet runs on the same body Filter.inlet returned, but
inlet strips the "[child_id:...]" prefix – so outlet could not re-read it and
wrote assessments to whatever _resolve_child_id returned (another mapped child
or DEFAULT_CHILD_ID). Inlet now records the resolved child on the body and
outlet reads it first, still subject to the F-014 authorization check."""
from tests.test_filter_prefix_authorization import (  # noqa: F401
    DEFAULT_CHILD, MY_CHILD, OTHER_CHILD, OWUI_USER, ff, filter_env)


def _round_trip(filter_env, content):
    body = {"messages": [{"role": "user", "content": content}], "chat_id": "chat-1"}
    user = {"id": OWUI_USER}
    body = ff.Filter().inlet(body, user)
    body["messages"].append({"role": "assistant", "content": "ответ"})
    ff.Filter().outlet(body, user)
    return body


def test_outlet_writes_to_the_prefix_child_inlet_resolved(filter_env):
    # The user is mapped to two children; _resolve_child_id returns the first.
    filter_env["conn"].cursor_obj._mappings[OWUI_USER] = [MY_CHILD, OTHER_CHILD]
    body = _round_trip(filter_env, f"[child_id:{OTHER_CHILD}] что такое луч?")
    assert body["messages"][-2]["content"] == "что такое луч?"  # prefix stripped for the model
    assert filter_env["updated"] == [OTHER_CHILD]


def test_outlet_without_prefix_keeps_the_mapped_child(filter_env):
    _round_trip(filter_env, "что такое луч?")
    assert filter_env["updated"] == [MY_CHILD]


def test_a_forged_body_child_is_still_authorization_checked(filter_env):
    body = {"messages": [{"role": "user", "content": "вопрос"}, {"role": "assistant", "content": "ответ"}],
            "chat_id": "chat-1", ff.RESOLVED_CHILD_KEY: OTHER_CHILD}
    ff.Filter().outlet(body, {"id": OWUI_USER})
    assert filter_env["updated"] == [MY_CHILD]
