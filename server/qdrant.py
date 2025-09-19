import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, CollectionStatus
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# client = QdrantClient(url="http://localhost:6333")


collection_name = "yelp_restaurants"
# try:
#     client.get_collection(collection_name=collection_name)
#     print(f"Collection '{collection_name}' already exists. Skipping creation.")
# except Exception:
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
#     )
#     print(f"Collection '{collection_name}' created.")

# file_path = "yelp_restaurants.json"
# file_path = "C:\Users\usejen_id\Desktop\llm_rag_hong\book-restaurant-agent\yelp\restaurants.json"
# with open(file_path, "r", encoding="utf-8") as f:
#     restaurants = json.load(f)    

# points = []
# for idx, restaurant in enumerate(restaurants):
#     # 필요한 필드가 없으면 건너뛰기
#     if not all(k in restaurant for k in ["name", "city", "categories", "stars", "attributes"]):
#         continue

#     # A. 모든 정보를 결합하여 벡터화할 description 생성
#     combined_description = (
#         f"This restaurant is named '{restaurant['name']}'. "
#         f"It is located in {restaurant['city']}. "
#         f"The cuisine type is {', '.join(restaurant['categories'])}. "
#         f"It has a star rating of {restaurant['stars']}. "
#     )
#     if 'text' in restaurant:
#         combined_description += f"A review states: '{restaurant['text']}'"
    
#     # B. description을 벡터로 변환
#     vector = model.encode(combined_description).tolist()

#     # C. Qdrant에 저장할 PointStruct 생성
#     point = PointStruct(
#         id=idx,  # 고유 ID
#         vector=vector,
#         payload=restaurant # 원본 메타데이터 저장
#     )
#     points.append(point)

# # 5. 모든 포인트를 일괄 업로드
# client.upsert(
#     collection_name=collection_name,
#     points=points
# )

# print(f"Successfully uploaded {len(points)} restaurant entries to Qdrant.")



def create_restaurant_description(data: dict) -> str:
    """
    Combines various restaurant data fields into a single, comprehensive text description for vectorization.
    It filters reviews based on star ratings to exclude less useful information.
    """
    description_parts = []
    
    # Add basic information
    name = data.get('name', 'This restaurant')
    city = data.get('city', 'a specific city')
    description_parts.append(f"The restaurant's name is '{name}', located in {city}.")
    
    # Add categories and attributes
    if data.get('categories'):
        categories = ", ".join(data['categories'])
        description_parts.append(f"It specializes in {categories} cuisine.")
        
    if data.get('ambiences'):
        ambiences = ", ".join(data['ambiences'])
        description_parts.append(f"The ambience is described as {ambiences}.")
        
    if data.get('good_for_kids'):
        description_parts.append("It is considered a good place for kids.")
    
    # Process tips and reviews
    positive_reviews = []
    negative_reviews = []
    
    # Combine tips and reviews into a single list
    all_reviews_and_tips = []
    if data.get('tips'):
        all_reviews_and_tips.extend([{"review": tip, "stars": None} for tip in data['tips']])
    if data.get('reviews'):
        all_reviews_and_tips.extend(data['reviews'])

    for item in all_reviews_and_tips:
        review_text = item.get('review')
        stars = item.get('stars')

        # Filter reviews by star rating
        if review_text and stars is not None:
            if stars >= 4:
                positive_reviews.append(review_text)
            elif stars <= 2:
                negative_reviews.append(review_text)
        elif review_text:  # For tips that have no star rating
            positive_reviews.append(review_text)

    # Add summarized review sections to the description
    if positive_reviews:
        positive_text = " ".join(positive_reviews[:5]) # Use only the first 5 positive reviews to avoid excessive length
        description_parts.append(f"Positive feedback highlights: {positive_text}")
    
    if negative_reviews:
        negative_text = " ".join(negative_reviews[:3]) # Use only the first 3 negative reviews
        description_parts.append(f"Negative feedback includes: {negative_text}")
    
    return " ".join(description_parts)


# 1. Initialize the embedding model and Qdrant client
# Replace with your own local or hosted model if different
print("1. Initializing SentenceTransformer model and Qdrant client...")
try:
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    client = QdrantClient(host="localhost", port=6333)
    print("Initialization successful.")
except Exception as e:
    print(f"Error initializing: {e}. Please ensure the model and Qdrant server are accessible.")
    exit()

# 2. Define collection and initialize it (recreates if it already exists)
collection_name = "yelp_restaurants"
print(f"\n2. Initializing collection '{collection_name}'. Deleting existing one if found...")
try:
    if client.get_collection(collection_name=collection_name).status != CollectionStatus.GREEN:
        client.delete_collection(collection_name=collection_name)
except Exception:
    pass # Ignore if the collection does not exist

if not client.collection_exists(collection_name=collection_name):

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
    )
print(f"Collection '{collection_name}' has been created.")


# 3. Load the JSON data
# file_path = "restaurants.json"
file_path = "C:\\Users\\usejen_id\\Desktop\\llm_rag_hong\\book-restaurant-agent\\yelp\\restaurants.json"
print(f"\n3. Loading data from '{file_path}'...")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        restaurants = json.load(f)
    print(f"Data loaded successfully. Found {len(restaurants)} entries.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please place it in the same directory.")
    exit()

# 4. Process data and prepare points for Qdrant
print("\n4. Processing data and preparing points...")
points = []
for idx, restaurant_data in enumerate(restaurants):
    # Create the combined text description
    combined_description = create_restaurant_description(restaurant_data)
    
    # Generate the vector embedding
    vector = model.encode(combined_description).tolist()
    
    # Prepare the PointStruct with vector and original data as payload
    point = PointStruct(
        id=idx,
        vector=vector,
        payload=restaurant_data
    )
    points.append(point)
print(f"Prepared {len(points)} points for upload.")

# 5. Upload all points to the Qdrant collection
print("\n5. Uploading points to Qdrant...")
client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points
)
print(f"Successfully uploaded {len(points)} restaurant entries.")

