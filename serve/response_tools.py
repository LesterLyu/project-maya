"""Codex tool/reasoning transport, following Basalt's Responses adapter design.

Namespaces become qualified template names; freeform inputs use a JSON string
argument internally. The public stream and replay retain their original types.
Reasoning replay is an opaque, base64 transport token, NOT encryption.
"""
from __future__ import annotations

import base64
import copy
import json
from dataclasses import replace

REPLAY_PREFIX = "maya.r1:"
HOSTED_TOOLS = {"web_search", "web_search_preview", "file_search", "computer_use_preview", "computer_use",
                "code_interpreter", "image_generation", "local_shell", "mcp"}


class ToolTransport:
    def __init__(self, req):
        self.original = req
        self.tools, self.names, self.aliases = [], {}, {}
        given = req.get("tools") if req.get("tools") is not None else []
        if not isinstance(given, list):
            raise ValueError("tools must be an array")
        items = req.get("input")
        extra = [t for item in items if isinstance(item, dict)
                 and item.get("type") in ("additional_tools", "tool_search_output")
                 for t in item.get("tools") or []] if isinstance(items, list) else []

        def add(tool, namespace=None, description="", depth=0):
            if not isinstance(tool, dict) or depth > 8:
                raise ValueError("invalid tool declaration")
            kind = tool.get("type", "function")
            # Like Basalt, expose client tools only; this server cannot execute
            # OpenAI-hosted tools. An explicitly forced hosted choice still fails.
            if kind in HOSTED_TOOLS:
                return
            name = tool.get("name")
            if kind == "tool_search":
                if tool.get("execution") not in (None, "client"):
                    raise ValueError("only client-executed tool_search is supported")
                name = "tool_search"
            if not isinstance(name, str) or not name:
                raise ValueError("a tool needs a nonempty name")
            qualified = f"{namespace}.{name}" if namespace else name
            desc = "\n\n".join(x for x in (description, tool.get("description")) if x)
            if kind == "namespace":
                members = tool.get("tools")
                if not isinstance(members, list):
                    raise ValueError("namespace.tools must be an array")
                for member in members:
                    add(member, qualified, desc, depth + 1)
                return
            if kind not in ("function", "custom", "tool_search"):
                raise ValueError(f"unsupported tool type: {kind!r}")
            if kind == "function" and set(tool) - {"type", "name", "description", "parameters", "strict", "defer_loading"}:
                raise ValueError("unsupported function tool parameter")
            flat = {k: v for k, v in tool.items() if k in ("description", "parameters", "strict")}
            flat.update(type="function", name=qualified)
            if desc:
                flat["description"] = desc
            if kind == "custom":
                fmt = tool.get("format") or {}
                if not isinstance(fmt, dict) or fmt.get("type", "text") not in ("text", "grammar"):
                    raise ValueError("unsupported custom tool format")
                if fmt.get("type") == "grammar":
                    definition = fmt.get("definition")
                    if not isinstance(definition, str):
                        raise ValueError("custom grammar needs a definition")
                    flat["description"] = desc + "\n\nThe raw input must follow this grammar:\n" + definition
                flat["parameters"] = {"type": "object", "properties": {
                    "input": {"type": "string", "description": "Raw tool input. Preserve code and newlines exactly."}},
                    "required": ["input"], "additionalProperties": False}
            spec = (namespace, name, kind)
            if qualified in self.names:
                if self.names[qualified] != spec or next(t for t in self.tools if t["name"] == qualified) != flat:
                    raise ValueError(f"conflicting tool declaration: {qualified}")
                return
            self.names[qualified] = spec
            self.tools.append(flat)

        for tool in [*given, *extra]:
            add(tool)
        shorts = {}
        for qualified, (namespace, name, _) in self.names.items():
            self.aliases[qualified] = qualified
            if namespace:
                self.aliases.setdefault(qualified.replace(".", "__"), qualified)
                self.aliases.setdefault(qualified.replace(".", "::"), qualified)
            shorts.setdefault(name, []).append(qualified)
        for name, candidates in shorts.items():
            if len(candidates) == 1:
                self.aliases.setdefault(name, candidates[0])

    def request(self):
        req = {**self.original, "tools": self.tools}
        choice = req.get("tool_choice")
        if isinstance(choice, dict) and choice.get("type") in ("function", "custom"):
            name = choice.get("name")
            if choice.get("namespace"):
                name = f"{choice['namespace']}.{name}"
            req["tool_choice"] = {"type": "function", "name": self.aliases.get(name, name)}
        if not isinstance(req.get("input"), list):
            return req
        items = []
        for raw in req["input"]:
            if not isinstance(raw, dict):
                items.append(raw)
                continue
            item = dict(raw)
            kind = item.get("type")
            if kind == "additional_tools":
                continue
            if kind == "reasoning" and item.get("encrypted_content"):
                token = item["encrypted_content"]
                if isinstance(token, str) and token.startswith(REPLAY_PREFIX):
                    if len(token) > 6 * 1024 * 1024:
                        raise ValueError("reasoning replay token exceeds its size limit")
                    try:
                        decoded = json.loads(base64.b64decode(token[len(REPLAY_PREFIX):], validate=True))
                        text = decoded["text"]
                        if not isinstance(text, str):
                            raise ValueError("reasoning text must be a string")
                    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                        raise ValueError("invalid Maya reasoning replay token") from exc
                    item.pop("encrypted_content")
                    if not item.get("content"):
                        item["content"] = [{"type": "reasoning_text", "text": text}]
            if kind in ("function_call", "custom_tool_call"):
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("tool calls need a nonempty name")
                if item.get("namespace"):
                    name = f"{item['namespace']}.{name}"
                item["name"] = self.aliases.get(name, name)
                item.pop("namespace", None)
                if kind == "custom_tool_call":
                    if not isinstance(item.get("input"), str):
                        raise ValueError("custom_tool_call.input must be a string")
                    item.update(type="function_call", arguments=json.dumps({"input": item.pop("input")}))
            elif kind == "custom_tool_call_output":
                item["type"] = "function_call_output"
            elif kind in ("tool_search_call", "tool_search_output"):
                if item.get("execution") not in (None, "client"):
                    raise ValueError("only client-executed tool_search is supported")
                if kind == "tool_search_call":
                    arguments = item.get("arguments", {})
                    item.update(type="function_call", name="tool_search", arguments=(
                        arguments if isinstance(arguments, str) else json.dumps(arguments)))
                else:
                    item.update(type="function_call_output", output=json.dumps(item.get("tools") or []))
            items.append(item)
        req["input"] = items
        return req

    def canonical_events(self, run):
        try:
            for kind, data in run:
                if kind == "event" and data.call is not None and data.call.name in self.aliases:
                    data = replace(data, call=replace(data.call, name=self.aliases[data.call.name]))
                    if data.kind == "tool_call" and data.complete and self.names[data.call.name][2] == "custom":
                        if not isinstance(data.call.arguments.get("input"), str):
                            raise ValueError("custom tool input must be a string")
                yield kind, data
        finally:
            run.close()

    def item(self, original):
        item = copy.deepcopy(original)
        if item.get("type") == "reasoning" and "reasoning.encrypted_content" in (self.original.get("include") or []):
            text = "".join(p.get("text", "") for p in item.get("content") or [])
            if text and item.get("status") == "completed":
                item["encrypted_content"] = REPLAY_PREFIX + base64.b64encode(json.dumps({"text": text}).encode()).decode()
        if item.get("type") != "function_call" or item.get("name") not in self.names:
            return item
        namespace, name, kind = self.names[item["name"]]
        item["name"] = name
        if namespace:
            item["namespace"] = namespace
        if kind in ("custom", "tool_search"):
            text = item.pop("arguments", "")
            value = json.loads(text) if text and item.get("status") == "completed" else {}
            if kind == "custom":
                raw = value.get("input", "")
                if not isinstance(raw, str):
                    raise ValueError("custom tool input must be a string")
                item.update(type="custom_tool_call", input=raw)
            else:
                item.pop("name", None)
                item.update(type="tool_search_call", execution="client", arguments=value)
        return item

    def response(self, original):
        return {**original, "tools": self.original.get("tools") or [],
                "tool_choice": self.original.get("tool_choice", "auto"),
                "output": [self.item(item) for item in original["output"]]}

    def events(self, source):
        sequence, calls = 0, {}
        try:
            for event in source:
                if event is None:
                    yield None
                    continue
                name, data = event
                if name == "response.output_item.added" and data["item"]["type"] == "function_call":
                    spec = self.names.get(data["item"]["name"])
                    if spec:
                        calls[data["item"]["id"]] = spec[2]
                kind = calls.get(data.get("item_id"))
                if name == "response.function_call_arguments.delta" and kind in ("custom", "tool_search"):
                    continue
                outgoing = []
                if name == "response.function_call_arguments.done" and kind in ("custom", "tool_search"):
                    if kind == "tool_search":
                        continue
                    value = json.loads(data["arguments"])
                    raw = value["input"]
                    if not isinstance(raw, str):
                        raise ValueError("custom tool input must be a string")
                    fields = {k: data[k] for k in ("item_id", "output_index")}
                    outgoing = [("response.custom_tool_call_input.delta", {**fields, "delta": raw}),
                                ("response.custom_tool_call_input.done", {**fields, "input": raw})]
                else:
                    data = dict(data)
                    if "item" in data:
                        data["item"] = self.item(data["item"])
                    if "response" in data:
                        data["response"] = self.response(data["response"])
                    outgoing = [(name, data)]
                for name, payload in outgoing:
                    yield name, {**payload, "type": name, "sequence_number": sequence}
                    sequence += 1
        finally:
            source.close()
