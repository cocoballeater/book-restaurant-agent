from typing_extensions import override
from typing import Callable, Any, AsyncGenerator, Optional
from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai.types import ModelContent

import json

from pydantic import BaseModel, Field

class SimpleAgent(BaseAgent):
    "Agent that wraps a user-provided function and executes it as part of the agent workflow."
    func: Callable[..., Any]
    input_key: Optional[str] = None

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        inputs = {self.input_key: ctx.user_content.parts[0].text} if self.input_key else {}
        output = await self._maybe_await(self.func(**inputs))

        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    content=ModelContent(json.dumps(output, ensure_ascii=False)))

    async def _maybe_await(self, value):
        if callable(getattr(value, "__await__", None)):
            return await value
        return value

def _handle_user_message(user_message: str) -> str:
    #pydantic schema output 정해주면 좋음
    if "추천" in user_message or "알려" in user_message:
        return [  {"type":"Message", "text":"강남역에서 점심 먹을만한 식당은 다음과 같습니다."},  {"type":"Restaurant Option", "title":"미소야", "id":"111"},  {"type":"Restaurant Option", "title":"홍콩반점", "id":"222"},  {"type":"Restaurant Option", "title":"양반", "id":"333"} ]
    elif "홍콩반점" in user_message:
        return [  {"type":"Reservation State", "title":"홍콩반점", "id":"222", "status":"생성"}, {"type":"Message", "text":"홍콩반점 예약을 도와드리겠습니다. 예약 날짜와 시간, 그리고 인원수를 알려주세요."} ]
    elif "모레" in user_message:
        return [  {"type":"Reservation State", "title":"홍콩반점", "id":"222", "status":"생성", "datetime":"2025-09-09 12:30", "persons": 4},  {"type":"Message", "text":"9월 9일 오후 12시 30분, 4명, 홍콩반점 예약을 진행할까요?"} ]
    elif user_message == "예약해줘.":
        return [  {"type":"Reservation State", "title":"홍콩반점", "id":"222", "status":"확정", "datetime":"2025-09-09 19:00", "persons": 4}, {"type":"Message", "text":"9월 9일 오후 7시, 4명, 홍콩반점 예약이 완료되었습니다."} ]

    return {"type":"Message", "text":"무엇을 도와드릴까요?"}
class Review(BaseModel):
    review:list[dict]
    tips:list[str] = Field(description = "List of tips for improving the restraurant experience")
root_agent = SimpleAgent(name="book_agent",
                         func=_handle_user_message,
                         input_key="user_message")
