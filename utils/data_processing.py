from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_chroma_document_from_mongo_document(mongo_document):
    legislation_id = mongo_document["_id"]
    legislation_name = mongo_document["name"]
    legislation_category = mongo_document["category"]
    legislation_department = mongo_document["department"]
    legislation_number_doc = mongo_document["numberDoc"]
    legislation_fields = mongo_document["fields"]
    
    legislation_content = mongo_document["content"]
    full_document = content_processing(legislation_content)
    
    try:
        metadata = {
            "id": legislation_id,
            "name": legislation_name,
            "category": legislation_category,
            "department": legislation_department,
            "numberDoc": legislation_number_doc,
            "fields": legislation_fields
        }
    except KeyError as e:
        raise ValueError(f"Missing expected key in mongo_document: {e}")
    
    return {
        "documents": full_document,
        "metadata": metadata
    }
    
# This function processes the content of a legislation document
# It extracts and concatenates the text from different sections of the document
# such as header, description, main content, and footer.
# The result is a single string that represents the full content of the legislation.
def content_processing(content):
    # Error handling for missing or invalid content structure
    if not isinstance(content, dict):
        raise ValueError("Content must be a dictionary with keys: header, description, mainContent, footer")
    for section in ["header", "description", "mainContent", "footer"]:
        if section in content and not isinstance(content[section], list):
            raise ValueError(f"Content section '{section}' must be a list")
    header = content.get("header", [])
    description = content.get("description", [])
    mainContent = content.get("mainContent", [])
    footer = content.get("footer", [])
    header_content = " ".join([content_node_processing(node) for node in header]) + "\n" 
    description_content = " ".join([content_node_processing(node) for node in description]) + "\n" 
    mainContent_content = " ".join([content_node_processing(node) for node in mainContent]) + "\n" 
    footer_content = " ".join([content_node_processing(node) for node in footer])
    full_document = header_content + description_content + mainContent_content + footer_content

    # Remove extra spaces and newlines
    full_document = " ".join(full_document.split())

    return full_document

# This function processes a single node in the content tree.
# It recursively concatenates the text from the node and its children.
# If the node has no content, it adds a space to maintain structure.
# The result is a string that represents the text of the node and its children.
def content_node_processing(node):
    value = node["value"]
    if not ("content" in node) or len(node["content"]) == 0:
        value += " "
    else:
        value += " ".join([content_node_processing(node_content) for node_content in node["content"]])
        
    return value