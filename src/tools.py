from retrieve import retrieve_documents

def Retrieval_Tool(query:str)->str:
    list_of_docs=[]
    result=retrieve_documents(query)
    if len(result)==0:
        list_of_docs.append("No related document has founded")
    for item in retrieve_documents(query):
        list_of_docs.append(item["payload"]["text"])
    return "\n".join(list_of_docs)

