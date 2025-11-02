from retrieve import retrieve_documents

def Retrieval_Tool(query:str)->str:
    list_of_docs=[]
    for item in retrieve_documents(query):
        list_of_docs.append(item["payload"]["text"])
    return "\n".join(list_of_docs)

print(Retrieval_Tool("where should I travell?"))