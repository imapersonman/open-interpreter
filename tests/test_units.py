import pytest

from interpreter.core.computer.computer import Computer
from interpreter.core.core import OpenInterpreter
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
