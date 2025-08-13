import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Get MongoDB URI from environment variable or use default
load_dotenv(dotenv_path=".env")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DBNAME = os.getenv("MONGO_DBNAME")

def get_db_from_mongo(mongo_url: str):
    # Kết nối đến MongoDB
    client = MongoClient(mongo_url)

    # Truy cập cơ sở dữ liệu
    collection = client[MONGO_DBNAME if MONGO_DBNAME!= None else "law_linking"]["laws"]

    # Trả về danh sách các văn bản json
    return collection, client

def get_legislation_by_query(query): 
    # Lấy collection từ hàm get_db_from_mongo
    collection, client = get_db_from_mongo(MONGO_URI)
    message = {"message": "", "data": ""}
    try:
        # Tìm tất cả tài liệu theo query
        legislations = collection.find(query)
        
        # Chuyển đổi kết quả thành danh sách
        legislations_list = [
            convert_document_from_db_to_available_json(doc) for doc in legislations
        ]
        
        if legislations_list:
            message = {"message": "success", "data": legislations_list}
        else:
            message = {"message": f"No legislations found with query {query}.", "data": []}
    except Exception as e:
        # Xử lý nếu có lỗi xảy ra
        message = {"message": f"An error occurred: {str(e)}"}

    client.close()
    
    return message

def convert_document_from_db_to_available_json(document):
    document["_id"] = str(document["_id"])
    # document["dateApproved"] = convert_datetime_to_string(document["dateApproved"])
    # document["createdAt"] = convert_datetime_to_string(document["createdAt"])
    # document["updatedAt"] = convert_datetime_to_string(document["updatedAt"])

    return document

# def convert_datetime_to_string(date_value):
#     if isinstance(date_value, datetime):
#         try:
#             # Try format with fractional seconds
#             return datetime.strftime(date_value, '%Y-%m-%dT%H:%M:%S.%fZ')
#         except ValueError:
#             try:
#                 # Try format without fractional seconds
#                 return datetime.strftime(date_value, '%Y-%m-%dT%H:%M:%SZ')
#             except ValueError:
#                 raise ValueError(f"Invalid value format for datetime: {date_value}")        


# if __name__ == "__main__":
#     # Ví dụ sử dụng hàm get_legislation_by_query
#     query = {"category": "Luật"}
#     result = get_legislation_by_query(query)
#     print(result["data"])