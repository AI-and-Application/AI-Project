from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI

classify_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """ Instructions for classification prompt here. """
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """ Instructions for answer prompt here. """
        ),
        HumanMessagePromptTemplate.from_template("{question}\n\nAdditional info: {additional_info}")
    ]
)

# Initialize LangChain models
classify_llm = ChatOpenAI(api_key="YOUR_OPENAI_API_KEY", model_name="gpt-4o-mini")
answer_llm = ChatOpenAI(api_key="YOUR_OPENAI_API_KEY", model_name="gpt-4o-mini")

# Chains
classify_chain = LLMChain(llm=classify_llm, prompt=classify_prompt)
answer_chain = LLMChain(llm=answer_llm, prompt=answer_prompt)
