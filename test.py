import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SubQueryCoverage, SubQueryUserInfoSimilarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOpenAI

async def main():
    # 初始化 LLM
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key="sk-cbaf36ff84eb4e17b301ecc7fd382d00"
    ))

    # SubQueryCoverage Example

    # sample = SingleTurnSample(
    #     user_input="Books about quantum mechanics and relativity",
    #     reference_contexts=["quantum mechanics books", "relativity theory books"],
    # )
    #
    # # 初始化 metric
    # coverage = SubQueryCoverage(llm=evaluator_llm)
    #
    # # 获取评分
    # score = await coverage.single_turn_ascore(sample)
    # print(f"SubQuery Coverage Score: {score}")

    ## SubQueryUserInfoSimilarity Example
    sample = SingleTurnSample(
        user_input="Where should I go for the best pizza in New York?",
        reference_contexts=[
            "best pizza in nyc",
            "top pizza places manhattan",
        ],
        user_profile="I live in Manhattan and love authentic Italian food",
    )

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    similarity = SubQueryUserInfoSimilarity(embeddings=embeddings)
    score = await similarity.single_turn_ascore(sample)
    print(f"SubQuery UserInfo Similarity Score: {score}")



# 执行主函数
if __name__ == "__main__":
    asyncio.run(main())
