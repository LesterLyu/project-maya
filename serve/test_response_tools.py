"""Codex tool transport, replay, and stream shapes over the actual Responses adapter."""
import copy
import json
import unittest

from serve.frontend import Event, ToolCall
from serve.responses import normalize_request, response_collect, response_events
from serve.response_tools import ToolTransport
from serve import test_responses as http_fixture

CUSTOM = {"type": "namespace", "name": "functions", "tools": [{
    "type": "custom", "name": "exec", "description": "Run JavaScript.",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /[\\s\\S]+/"}}]}
FUNCTION = {"type": "namespace", "name": "functions", "tools": [{
    "type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}]}
SEARCH = {"type": "tool_search", "execution": "client", "parameters": {"type": "object"}}


def stream(events):
    for event in events:
        yield "event", event
    yield "done", {"finish": "stop", "prompt_tokens": 10, "completion_tokens": 4}


class ToolCompatibility(unittest.TestCase):
    def test_namespace_and_custom_schema_do_not_mutate_request(self):
        req = {"input": "hi", "tools": [CUSTOM, FUNCTION]}
        before = copy.deepcopy(req)
        normalized = normalize_request(req)
        self.assertEqual([t["name"] for t in normalized["tools"]], ["functions.exec", "functions.lookup"])
        self.assertIn("grammar", normalized["tools"][0]["description"])
        self.assertEqual(req, before)

    def test_custom_round_trip_preserves_multiline_code_and_call_id(self):
        code = 'print("hello")\nprint("\\n")\n'
        req = {"input": "hi", "tools": [CUSTOM]}
        call = ToolCall("functions.exec", {"input": code}, id="call-code")
        events = list(response_events("m", req, stream([
            Event("tool_start", call=call),
            Event("tool_args", text=json.dumps({"input": code})[:9], call=call),
            Event("tool_args", text=json.dumps({"input": code})[9:], call=call),
            Event("tool_call", call=call),
        ])))
        names = [name for name, _ in events]
        self.assertIn("response.custom_tool_call_input.delta", names)
        self.assertNotIn("response.function_call_arguments.delta", names)
        self.assertEqual([data["sequence_number"] for _, data in events], list(range(len(events))))
        item = events[-1][1]["response"]["output"][0]
        self.assertEqual((item["type"], item["namespace"], item["name"], item["input"], item["call_id"]),
                         ("custom_tool_call", "functions", "exec", code, "call-code"))
        replay = normalize_request({"tools": [CUSTOM], "input": [
            {"role": "user", "content": "hi"}, item,
            {"type": "custom_tool_call_output", "call_id": "call-code", "output": "done"}]})
        self.assertEqual(json.loads(replay["messages"][1]["tool_calls"][0]["function"]["arguments"]), {"input": code})
        self.assertEqual(replay["messages"][2]["content"], "done")

    def test_function_namespace_and_unambiguous_alias(self):
        call = ToolCall("functions__lookup", {}, id="c")
        req = {"input": "hi", "tools": [FUNCTION]}
        out = response_collect(response_events("m", req, stream([Event("tool_call", call=call)])))
        item = out["output"][0]
        self.assertEqual((item["namespace"], item["name"], item["arguments"]), ("functions", "lookup", "{}"))

    def test_client_tool_search_and_discovered_tool_replay(self):
        out = response_collect(response_events("m", {"input": "hi", "tools": [SEARCH]}, stream([
            Event("tool_call", call=ToolCall("tool_search", {"query": "lookup"}, id="search-1"))])))
        call = out["output"][0]
        self.assertEqual(call["type"], "tool_search_call")
        self.assertEqual(call["arguments"], {"query": "lookup"})
        req = {"tools": [SEARCH], "input": [{"role": "user", "content": "hi"}, call,
            {"type": "tool_search_output", "execution": "client", "call_id": "search-1", "tools": [FUNCTION]}]}
        normalized = normalize_request(req)
        self.assertIn("functions.lookup", [tool["name"] for tool in normalized["tools"]])
        self.assertEqual(json.loads(normalized["messages"][-1]["content"]), [FUNCTION])

    def test_reasoning_replay_is_opaque_not_claimed_encryption(self):
        req = {"input": "hi", "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}
        out = response_collect(response_events("m", req, stream([Event("reasoning", text="think"),
                                                                Event("content", text="answer")])))
        reasoning = out["output"][0]
        self.assertTrue(reasoning["encrypted_content"].startswith("maya.r1:"))
        reasoning["content"] = []
        replay = normalize_request({"input": [reasoning]})
        self.assertEqual(replay["messages"][0]["reasoning_content"], "think")
        with self.assertRaises(ValueError):
            normalize_request({"input": [{"type": "reasoning", "encrypted_content": "foreign"}]})

    def test_summary_hints_preserve_reasoning_and_replay(self):
        for summary in (None, "auto", "concise", "detailed"):
            with self.subTest(summary=summary):
                req = {"input": "hi", "reasoning": {"effort": "low", "summary": summary},
                       "include": ["reasoning.encrypted_content"]}
                self.assertEqual(normalize_request(req)["reasoning"], req["reasoning"])
                out = response_collect(response_events("m", req, stream([
                    Event("reasoning", text="think"), Event("content", text="answer")])))
                item = out["output"][0]
                self.assertEqual(item["content"], [{"type": "reasoning_text", "text": "think"}])
                self.assertEqual(item["summary"], [])
                self.assertTrue(item["encrypted_content"].startswith("maya.r1:"))
                self.assertEqual(normalize_request({"input": [item]})["messages"][0]["reasoning_content"], "think")
        for summary in ("invalid", True, 42, [], {}):
            with self.subTest(summary=summary), self.assertRaisesRegex(ValueError, "reasoning.summary"):
                normalize_request({"input": "hi", "reasoning": {"summary": summary}})

    def test_stored_response_has_public_custom_shape(self):
        saved = []
        req = {"input": "hi", "tools": [CUSTOM]}
        out = response_collect(response_events("m", req, stream([
            Event("tool_call", call=ToolCall("functions.exec", {"input": "1+1"}))]), persist=saved.append))
        self.assertEqual(saved[0], out)
        self.assertEqual(saved[0]["tools"], [CUSTOM])

    def test_invalid_custom_input_fails_without_exposing_a_completed_call(self):
        req = {"input": "hi", "tools": [CUSTOM]}
        out = response_collect(response_events("m", req, stream([
            Event("tool_call", call=ToolCall("functions.exec", {"input": 123}))])))
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["output"], [])

    def test_invalid_tools_and_conflicting_discovery_fail(self):
        for tools in ({}, [{"type": "unknown_tool"}], [{"type": "namespace", "name": "n", "tools": {}}]):
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                normalize_request({"input": "hi", "tools": tools})
        other = copy.deepcopy(FUNCTION)
        other["tools"][0]["description"] = "changed"
        with self.assertRaises(ValueError):
            ToolTransport({"tools": [FUNCTION, other]})

    def test_codex_optional_fields_and_hosted_tool_boundary(self):
        out = normalize_request({"input": "hi", "tools": [CUSTOM, {"type": "web_search"}],
                                 "text": {"verbosity": "low"}, "reasoning": {"effort": "high", "context": "all_turns"}})
        self.assertEqual([t["name"] for t in out["tools"]], ["functions.exec"])
        with self.assertRaises(ValueError):
            normalize_request({"input": "hi", "tools": [{"type": "web_search"}],
                               "tool_choice": {"type": "web_search"}})


class CodexHttp(unittest.TestCase):
    setUp = http_fixture.ResponseCase.setUp
    tearDown = http_fixture.ResponseCase.tearDown
    request = http_fixture.ResponseCase.request
    post = http_fixture.ResponseCase.post
    stream = http_fixture.ResponseCase.stream
    script = http_fixture.ResponseCase.script

    def test_code_mode_stream_and_tool_result_continue(self):
        code = "const x = 1;\ntext(x);"
        self.script(['</think><tool_call>functions.exec<arg_key>input</arg_key><arg_value>' + code +
                     '</arg_value></tool_call>', '</think>Done'])
        events = self.stream(tools=[CUSTOM], reasoning={"effort": "low", "summary": "auto"})
        out = events[-1]["response"]
        self.assertEqual(out["status"], "completed")
        call = out["output"][0]
        self.assertEqual((call["type"], call["input"]), ("custom_tool_call", code))
        status, out = self.post(tools=[CUSTOM], reasoning={"effort": "low", "summary": "auto"},
                               input=[{"role": "user", "content": "Run code"}, call,
            {"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "1"}])
        self.assertEqual((status, out["output"][0]["content"][0]["text"]), (200, "Done"))

    def test_json_schema_with_custom_tools_and_verbosity(self):
        self.script('{"approved":true}')
        status, out = self.post(tools=[CUSTOM], text={"verbosity": "low", "format": {
            "type": "json_schema", "name": "approval", "strict": True,
            "schema": {"type": "object", "properties": {"approved": {"type": "boolean"}},
                       "required": ["approved"], "additionalProperties": False}}})
        self.assertEqual((status, out["status"]), (200, "completed"))
        self.assertEqual(json.loads(out["output"][0]["content"][0]["text"]), {"approved": True})


if __name__ == "__main__":
    unittest.main()
