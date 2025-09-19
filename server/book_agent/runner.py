import dotenv
dotenv.load_dotenv()

from agent import root_agent
from agent import Review

import asyncio
from google.adk.runners import InMemoryRunner
from google.genai.types import UserContent

import json 
import os
async def main():
    runner = InMemoryRunner(agent=root_agent, app_name=root_agent.name)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id='user1')
    
    # 1. restaurants.json 파일 경로 설정 및 로드
    # OSError를 방지하기 위해 os.path.join 사용
    restaurants_json_path = os.path.join(
        "C:\\Users\\usejen_id\\Desktop\\llm_rag_hong\\book-restaurant-agent\\yelp",
        "restaurants.json"
    )
    
    try:
        with open(restaurants_json_path, "r", encoding="utf-8") as f:
            restaurants = json.load(f)
        print(f"Data loaded successfully. Found {len(restaurants)} entries.")
    except FileNotFoundError:
        print(f"Error: The file '{restaurants_json_path}' was not found. Please check the path.")
        return

    # 2. agent를 사용하여 tips를 10개의 특징으로 분류
    for i, restaurant in enumerate(restaurants):
        if not restaurant.get('tips'):
            continue # tips가 없는 경우 건너뛰기
            
        print(f"Processing restaurant {i+1}/{len(restaurants)}: {restaurant.get('name', 'N/A')}")

        # agent에 보낼 데이터 준비
        input_data = {
            "reviews": restaurant.get('reviews', []),
            "tips": restaurant['tips']
        }
        review_json = json.dumps(input_data, ensure_ascii=False)

        # agent 실행 및 응답 수신
        # async for 대신 await를 사용해 최종 응답만 받습니다.
        event_iterator = runner.run(user_id=session.user_id, session_id=session.id, new_message=UserContent(review_json))
        
        # 비동기 이터레이터의 첫 번째(이자 유일한) 이벤트 가져오기
        try:
            event = event_iterator.__next__()
            if event.is_final_response():
                try:
                    agent_output = json.loads(event.content.parts[0].text)
                    restaurant['tips_new'] = agent_output.get('tips_new', [])
                    print(f"-> Classified {len(restaurant['tips'])} tips into {len(restaurant['tips_new'])} features.")
                except (json.JSONDecodeError, IndexError) as e:
                    print(f"-> Failed to process agent response for restaurant {restaurant.get('name')}: {e}")
                    restaurant['tips_new'] = ["Failed to classify tips."]
        except StopAsyncIteration:
            print("-> No response received from agent.")
            restaurant['tips_new'] = ["No response received from agent."]

    # 3. restaurants_desc.json으로 저장
    restaurants_desc_path = os.path.join(
        os.path.dirname(restaurants_json_path),
        "restaurants_desc.json"
    )
    with open(restaurants_desc_path, "w", encoding="utf-8") as f:
        json.dump(restaurants, f, ensure_ascii=False, indent=4)
    print(f"\nProcessing complete. Saved data to '{restaurants_desc_path}'.")

if __name__ == "__main__":
    asyncio.run(main())
