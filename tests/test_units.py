from collections import deque
from typing import Deque, List
import pytest

from interpreter.core.computer.computer import Computer
from interpreter.core.core import OpenInterpreter
from interpreter.core.llm.base import BaseLlm
from interpreter.core.llm.llm import Llm


@pytest.mark.unit
def test_default_constructor_no_smoke():
    """
    This test is a smoke test.  Its just here to make sure nothing bad happens when we try to
    construct an OpenInterpreter object.  A surprising amount of bugs can be caught early by just
    running this test.
    """
    OpenInterpreter()


@pytest.mark.unit
def test_non_default_constructor_no_smoke():
    """
    This test is a smoke test like the one above, but it tries to avoid using default values.  If
    defaults change in the future, this test doesn't necessarily need to change to keep working.
    """
    OpenInterpreter(
        messages=[
            {
                "role": "user",
                "type": "message",
                "content": "why was I put on this earth?",
            },
            {"role": "assistant", "type": "message", "content": "im a computer chill"},
        ],
        offline=True,
        auto_run=True,
        verbose=True,
        debug=True,
        max_output=1000,
        safe_mode="auto",
        shrink_images=True,
        disable_telemetry=True,
        in_terminal_interface=True,
        multi_line=True,
        force_task_completion=True,
        force_task_completion_message="please run this code 🥺",
        force_task_completion_breakers=[],
        conversation_history=False,
        conversation_filename="conversation.json",
        conversation_history_path=".",
        os=True,
        speak_messages=True,
        llm=None,  # still the default value but its fine.
        system_message="u",
        custom_instructions="bu",
        computer=None,  # still the default value but its fine.
        sync_computer=True,
        import_computer_api=True,
        skills_path=None,  # still the default value but its fine.
        import_skills=True,
    )


@pytest.mark.unit
def test_constructor_syncs_computer_settings():
    """
    Tests that the common settings between interpreter and computer (import_computer_api,
    skills_path, and import_skills) are the same between and interpreter and computer.  This is
    tested for two different OpenInterpreter instances.

    QUESTION: should these settings really be duplicated?
    """
    i1 = OpenInterpreter(
        import_computer_api=True, skills_path="fake/path", import_skills=True
    )
    assert i1.import_computer_api == i1.computer.import_computer_api
    assert i1.skills_path == i1.computer.skills.path
    assert i1.import_skills == i1.computer.import_skills

    i2 = OpenInterpreter(
        import_computer_api=False, skills_path="something/else", import_skills=False
    )
    assert i2.import_computer_api == i2.computer.import_computer_api
    assert i2.skills_path == i2.computer.skills.path
    assert i2.import_skills == i2.computer.import_skills


@pytest.mark.unit
def test_computer_and_llm_none_to_constructed():
    """
    Tests to make sure that setting computer and llm to None during OpenInterpreter construction
    sets each to actual instances of Computer and Llm.
    """
    i = OpenInterpreter(computer=None, llm=None)
    assert isinstance(i.computer, Computer)
    assert isinstance(i.llm, Llm)


class MockLlm(BaseLlm):
    def __init__(self):
        super().__init__(model="mock")
        self._queue: Deque[List] = deque()
        self._last_run_input = None

    def run(self, messages):
        if len(self._queue) == 0:
            raise RuntimeError("MockLlm expects another call to run!")
        self._last_run_input = messages
        yield from self._queue.popleft()

    @property
    def last_run_input(self):
        return self._last_run_input

    def add_response(self, message):
        self._queue.append(message)


def assert_gen(actual_it, expected_it):
    for index, (expected, actual) in enumerate(zip(expected_it, actual_it)):
        assert actual == expected, f"actual not equal to expected at index {index}"


@pytest.mark.unit
def test_streaming_interaction_message_only():
    llm = MockLlm()
    interpreter = OpenInterpreter(llm=llm)

    llm.add_response(
        [{"role": "assistant", "type": "message", "content": "anything at all"}]
    )

    assert_gen(
        interpreter.chat("say anything at all", display=False, stream=True),
        [
            {"role": "assistant", "type": "message", "start": True},
            {
                "role": "assistant",
                "type": "message",
                "content": "anything at all",
            },
            {"role": "assistant", "type": "message", "end": True},
        ],
    )

    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": "say anything at all"},
        {"role": "assistant", "type": "message", "content": "anything at all"},
    ]


@pytest.mark.unit
def test_streaming_interaction_code_only():
    llm = MockLlm()
    interpreter = OpenInterpreter(llm=llm)

    llm.add_response(
        [
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "'this is just' + ' some code'",
            }
        ],
    )
    # remember: code is never the last thing run!
    llm.add_response(
        [{"role": "assistant", "type": "message", "content": "we're done!"}]
    )

    assert_gen(
        interpreter.chat("please run code", display=False, stream=True),
        [
            {"role": "assistant", "type": "code", "format": "python", "start": True},
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "'this is just' + ' some code'",
            },
            {"role": "assistant", "type": "code", "format": "python", "end": True},
            {
                "role": "computer",
                "type": "confirmation",
                "format": "execution",
                "content": {
                    "type": "code",
                    "format": "python",
                    "content": "'this is just' + ' some code'",
                },
            },
            {"role": "computer", "type": "console", "start": True},
            {
                "role": "computer",
                "type": "console",
                "format": "active_line",
                "content": 1,
            },
            {
                "role": "computer",
                "type": "console",
                "format": "output",
                "content": "'this is just some code'",
            },
            {
                "role": "computer",
                "type": "console",
                "format": "active_line",
                "content": None,
            },
            {"role": "computer", "type": "console", "end": True},
            {"role": "assistant", "type": "message", "start": True},
            {"role": "assistant", "type": "message", "content": "we're done!"},
            {"role": "assistant", "type": "message", "end": True},
        ],
    )

    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": "please run code"},
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "'this is just' + ' some code'",
        },
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "'this is just some code'",
        },
        {"role": "assistant", "type": "message", "content": "we're done!"},
    ]


@pytest.mark.unit
def test_streaming_interaction():
    """
    This test ensures that all the "fluff" (streaming boundary flags, execution confirmation, etc.)
    is added without issue in the simple streaming interaction seen in the docs:
    https://docs.openinterpreter.com/guides/streaming-response
    """

    llm = MockLlm()
    interpreter = OpenInterpreter(llm=llm)

    llm.add_response(
        [
            {"role": "assistant", "type": "code", "format": "python", "content": "3"},
            {"role": "assistant", "type": "code", "format": "python", "content": " /"},
            {"role": "assistant", "type": "code", "format": "python", "content": " "},
            {"role": "assistant", "type": "code", "format": "python", "content": "2"},
        ]
    )

    llm.add_response(
        [
            {"role": "assistant", "type": "message", "content": "The"},
            {"role": "assistant", "type": "message", "content": " result"},
            {"role": "assistant", "type": "message", "content": " of"},
            {"role": "assistant", "type": "message", "content": " the"},
            {"role": "assistant", "type": "message", "content": " division"},
            {"role": "assistant", "type": "message", "content": " "},
            {"role": "assistant", "type": "message", "content": "3"},
            {"role": "assistant", "type": "message", "content": "/"},
            {"role": "assistant", "type": "message", "content": "2"},
            {"role": "assistant", "type": "message", "content": " is"},
            {"role": "assistant", "type": "message", "content": " approximately"},
            {"role": "assistant", "type": "message", "content": " "},
            {"role": "assistant", "type": "message", "content": "1"},
            {"role": "assistant", "type": "message", "content": "."},
            {"role": "assistant", "type": "message", "content": "5"},
            {"role": "assistant", "type": "message", "content": "."},
        ]
    )

    assert_gen(
        interpreter.chat("do something cool please", display=False, stream=True),
        [
            {"role": "assistant", "type": "code", "format": "python", "start": True},
            {"role": "assistant", "type": "code", "format": "python", "content": "3"},
            {"role": "assistant", "type": "code", "format": "python", "content": " /"},
            {"role": "assistant", "type": "code", "format": "python", "content": " "},
            {"role": "assistant", "type": "code", "format": "python", "content": "2"},
            {"role": "assistant", "type": "code", "format": "python", "end": True},
            {
                "role": "computer",
                "type": "confirmation",
                "format": "execution",
                "content": {"type": "code", "format": "python", "content": "3 / 2"},
            },
            {"role": "computer", "type": "console", "start": True},
            {
                "role": "computer",
                "type": "console",
                "format": "active_line",
                "content": 1,
            },
            {
                "role": "computer",
                "type": "console",
                "format": "output",
                "content": "1.5",
            },
            {
                "role": "computer",
                "type": "console",
                "format": "active_line",
                "content": None,
            },
            {"role": "computer", "type": "console", "end": True},
            {"role": "assistant", "type": "message", "start": True},
            {"role": "assistant", "type": "message", "content": "The"},
            {"role": "assistant", "type": "message", "content": " result"},
            {"role": "assistant", "type": "message", "content": " of"},
            {"role": "assistant", "type": "message", "content": " the"},
            {"role": "assistant", "type": "message", "content": " division"},
            {"role": "assistant", "type": "message", "content": " "},
            {"role": "assistant", "type": "message", "content": "3"},
            {"role": "assistant", "type": "message", "content": "/"},
            {"role": "assistant", "type": "message", "content": "2"},
            {"role": "assistant", "type": "message", "content": " is"},
            {"role": "assistant", "type": "message", "content": " approximately"},
            {"role": "assistant", "type": "message", "content": " "},
            {"role": "assistant", "type": "message", "content": "1"},
            {"role": "assistant", "type": "message", "content": "."},
            {"role": "assistant", "type": "message", "content": "5"},
            {"role": "assistant", "type": "message", "content": "."},
            {"role": "assistant", "type": "message", "end": True},
        ],
    )

    # let's also make sure the messages are accumulated properly
    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": "do something cool please"},
        {"role": "assistant", "type": "code", "format": "python", "content": "3 / 2"},
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "1.5",
        },
        {
            "role": "assistant",
            "type": "message",
            "content": "The result of the division 3/2 is approximately 1.5.",
        },
    ]


@pytest.mark.unit
def test_streaming_interaction_multiple_code_blocks():
    """
    This test checks for the same fluff as the last, but also sees what happens when multiple code
    blocks are proposed.  The examples are simplified to shorten the length of the test.
    """

    llm = MockLlm()
    interpreter = OpenInterpreter(llm=llm)

    llm.add_response(
        [
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "'first block output'",
            },
            {
                "role": "assistant",
                "type": "message",
                "content": "the above content does something idk",
            },
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "'second block output'",
            },
        ]
    )

    llm.add_response(
        [{"role": "assistant", "type": "message", "content": "anything at all"}]
    )

    # list(interpreter.chat("give me lots of code please", display=False))
    # print_yields(interaction.chat("give me lots of code"))
    assert_gen(
        interpreter.chat("give me lots of code please", display=False, stream=True),
        [
            {"role": "assistant", "type": "code", "format": "python", "start": True},
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "'first block output'",
            },
            {"role": "assistant", "type": "code", "format": "python", "end": True},
            {"role": "assistant", "type": "message", "start": True},
            {
                "role": "assistant",
                "type": "message",
                "content": "the above content does something idk",
            },
            {"role": "assistant", "type": "message", "end": True},
            {"role": "assistant", "type": "code", "format": "python", "start": True},
            {
                "role": "assistant",
                "type": "code",
                "format": "python",
                "content": "'second block output'",
            },
            {"role": "assistant", "type": "code", "format": "python", "end": True},
            {
                "role": "computer",
                "type": "confirmation",
                "format": "execution",
                "content": {
                    "type": "code",
                    "format": "python",
                    "content": "'second block output'",
                },
            },
            {"role": "computer", "type": "console", "start": True},
            {
                "role": "computer",
                "type": "console",
                "format": "active_line",
                "content": 1,
            },
            {
                "role": "computer",
                "type": "console",
                "format": "output",
                "content": "'second block output'",
            },
            {
                "role": "computer",
                "type": "console",
                "format": "active_line",
                "content": None,
            },
            {"role": "computer", "type": "console", "end": True},
            # it looks like the main loop expects the assistant to respond after the console prints.
            {"role": "assistant", "type": "message", "start": True},
            {"role": "assistant", "type": "message", "content": "anything at all"},
            {"role": "assistant", "type": "message", "end": True},
        ],
    )

    # let's also make sure the messages are accumulated properly
    assert interpreter.messages == [
        {"role": "user", "type": "message", "content": "give me lots of code please"},
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "'first block output'",
        },
        {
            "role": "assistant",
            "type": "message",
            "content": "the above content does something idk",
        },
        {
            "role": "assistant",
            "type": "code",
            "format": "python",
            "content": "'second block output'",
        },
        {
            "role": "computer",
            "type": "console",
            "format": "output",
            "content": "'second block output'",
        },
        {"role": "assistant", "type": "message", "content": "anything at all"},
    ]
