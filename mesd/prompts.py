"""Exact Open-o3 student prompt contract.

Do not replace these messages with RLSD's generic ``unified.jinja`` template:
that template changes both the role structure and the output tag convention.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schema import STGRRecord, coerce_record


SYSTEM_PROMPTS = {
    "visual QA": "A conversation between user and assistant. The user provides an image and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When referring to particular objects in the reasoning process, the assistant must localize the object with bounding box coordinates between <box> and </box>. The answer must strictly follow the following format:`<obj>object_name</obj><box>bounding_box</box>'.",
    "temporal-spatial free-form QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed.",
    "temporal QA": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant determines the precise time period that answers the question. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'.The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s'.",
    "temporal QA (MCQ)": "A conversation between user and assistant. The user provides a video and a multiple-choice question, and the Assistant determines the precise time period that answers the question and selects the correct option. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. When mentioning time during the reasoning process, the assistant must use the format: `<t>time_in_seconds</t>s'. The answer must strictly follow the following format: `From <t>start_time</t>s to <t>end_time</t>s.\nCorrect Option: [ONLY THE LETTER]'.",
    "General video QA MCQ": "A conversation between user and assistant. The user provides a video and asks a multiple-choice question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. Only output the correct option in the <answer> </answer> section.",
    "General video QA Free-form": "A conversation between user and assistant. The user provides a video and asks a question, and the Assistant solves it. The assistant MUST first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. All reasoning must be grounded in visual evidence from the video. When you mention any related object, person, or specific visual element in the reasoning process, you must strictly follow the following format: `<obj>object_name</obj><box>bounding_box</box>at<t>time_in_seconds</t>s`. The answer part only requires a text response; tags like <obj>, <box>, <t> are not needed.",
}


def build_student_messages(
    row: STGRRecord | Mapping[str, Any],
    *,
    media_path: str | None = None,
) -> list[dict[str, Any]]:
    """Build the two-message Open-o3 student view.

    The system role is semantically important.  ``media_path=None`` reproduces
    the released loader's placeholder-only message; a path can be attached for
    processors that decode directly from the message.
    """

    record = coerce_record(row)
    media_type = "image" if record.task == "visual QA" else "video"
    media: dict[str, Any] = {"type": media_type}
    if media_path is not None:
        media[media_type] = str(media_path)

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPTS[record.task]}],
        },
        {
            "role": "user",
            "content": [media, {"type": "text", "text": record.question}],
        },
    ]


def apply_student_chat_template(processor, row: STGRRecord | Mapping[str, Any]) -> str:
    """Serialize student messages with the model checkpoint's chat template."""

    return processor.apply_chat_template(
        build_student_messages(row),
        tokenize=False,
        add_generation_prompt=True,
    )
