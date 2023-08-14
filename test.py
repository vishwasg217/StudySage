from langchain.schema import SystemMessage 
from langchain.memory import ConversationBufferWindowMemory

system_message = SystemMessage(content="Hello world!")

memory = ConversationBufferWindowMemory(
    k=5,
    extra_messages=[system_message]  
)

print(memory.construct_prompt(...))
# Should see "Hello world!" after history

chain = MyChain(memory=memory) 
response = chain.predict(...) 
# SystemMessage should be in prompt here