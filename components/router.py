from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda 
from langchain_ollama import ChatOllama

def remove_non_alphabetic(s: str) -> str:
    """ Remove non-alphabetic characters from a string"""
    return ''.join(filter(str.isalpha, s))

def route_tool(query: str, llm: ChatOllama) -> str:
    """
    Route query to correct process.
    """
    template = """
    Bạn là một trợ lý pháp lý cho người dùng với công cụ truy vấn các văn bản pháp luật Việt Nam \n
    Từ câu hỏi của người dùng, hãy trả lời rằng có cần thiết để sử dụng bộ công cụ truy vấn không: {question} \n
    Đầu ra CHỈ có duy nhất một từ "có" hoặc "không", KHÔNG có bất kỳ lời giải thích hay từ nào khác. Nếu câu hỏi liên quan đến các văn bản pháp luật, quy định, điều luật, thủ tục hành chính, hãy trả lời "có", ngược lại trả lời "không".
    """
    prompt_router = ChatPromptTemplate.from_template(template)
    RemoveNonAlphabetic = RunnableLambda(remove_non_alphabetic)
    
    route_chain = (
        prompt_router 
        | llm
        | StrOutputParser() 
        | RemoveNonAlphabetic
    )
    response = route_chain.invoke({"question": query})
    print("Router:", response)
    if response and response.strip().lower() == "có":
        return "có"
    else:
        return "không"