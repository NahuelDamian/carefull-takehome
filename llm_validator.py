import os
import time
import pandas as pd
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from heuristics import TransactionAnalyzer

# --- Provider configuration -------------------------------------------------
# Model names live here (overridable via environment) instead of being hardcoded
# in the constructor, so swapping a deprecated model is a one-line change.
PROVIDER_CONFIG = {
    "groq": {
        "default_model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
        "api_key_env": "GROQ_API_KEY",
        # Groq's free tier enforces aggressive rate limits, so we pace requests.
        "request_delay_sec": 1.5,
    },
    "openrouter": {
        "default_model": os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        "api_key_env": "OPENROUTER_API_KEY",
        "request_delay_sec": 0.5,
        "base_url": "https://openrouter.ai/api/v1",
    },
}

# 1. Define the structure
class RiskAssessment(BaseModel):
    risk_level: str = Field(description="Classification: 'high', 'medium', or 'low'")
    reasoning: str = Field(description="Detailed explanation based on user profile.")
    confidence_score: int = Field(description="Confidence score from 1 to 100")

class FraudValidatorAgent:
    def __init__(self, provider="groq", model_name=None):
        """
        Args:
            provider: Key of PROVIDER_CONFIG ('groq' or 'openrouter').
            model_name: Optional override. Defaults to the provider's configured model.
        """
        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {', '.join(PROVIDER_CONFIG)}"
            )

        config = PROVIDER_CONFIG[provider]
        api_key = os.environ.get(config["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"Missing {config['api_key_env']}. Export it before running the validator."
            )

        self.provider = provider
        self.model_name = model_name or config["default_model"]
        self.request_delay_sec = config["request_delay_sec"]

        if provider == "groq":
            self.llm = ChatGroq(model=self.model_name, temperature=0.1)
        else:
            self.llm = ChatOpenAI(
                base_url=config["base_url"],
                api_key=api_key,
                model=self.model_name,
                temperature=0.1
            )
            
        # 2. We use PydanticOutputParser (it is provider-agnostic and avoids OpenRouter error)
        self.parser = PydanticOutputParser(pydantic_object=RiskAssessment)
        
        # 3. We insert the format instructions directly into the prompt
        self.prompt = PromptTemplate(
            template="""You are an expert fraud detection AI for 'Carefull', protecting older adults. Balance finding scams with avoiding false alarms for active seniors.
            
USER PROFILE:
Name: {name} | Age: {age} | Income: ${income} | Bio: {bio}

TRANSACTION:
Date: {date} | Amount: ${amount} | Type: {type} | Description: {description}

Determine risk ('high'/'medium'/'low'), reasoning, and confidence.
{format_instructions}
""",
            input_variables=["name", "age", "income", "bio", "date", "amount", "type", "description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.llm | self.parser

    def evaluate_transactions(self, suspicious_df):
        """Sends each heuristically-flagged transaction to the LLM and collects its verdict."""
        results = []
        print(f"\nStarting evaluation with {self.provider.upper()} ({self.model_name})...")
        
        for index, row in suspicious_df.iterrows():
            start_time = time.time()
            try:
                assessment = self.chain.invoke({
                    "name": row['name'], "age": row['age'], "income": row['monthly_income'],
                    "bio": row['bio'], "date": row['date'], "amount": row['amount'],
                    "type": row['type'], "description": row['raw_description']
                })
                latency = round(time.time() - start_time, 2)
                
                results.append({
                    "transaction_id": row['transaction_id'],
                    "user_id": row['user_id'],
                    "risk_level": assessment.risk_level,
                    "reasoning": assessment.reasoning,
                    "confidence": assessment.confidence_score,
                    "latency_sec": latency,
                    "provider": self.provider
                })
                # Respect the provider's rate limits between calls
                time.sleep(self.request_delay_sec)
                
            except Exception as e:
                print(f"Error in tx {row['transaction_id']}: {e}")
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = TransactionAnalyzer("datasets/transactions.csv", "datasets/user_profiles.csv")
    suspicious_tx = analyzer.run_pipeline()
    
    # 1. Evaluate with Groq (model resolved from PROVIDER_CONFIG / GROQ_MODEL)
    agent_groq = FraudValidatorAgent(provider="groq")
    results_groq = agent_groq.evaluate_transactions(suspicious_tx)
    results_groq.to_csv(f"{output_dir}/results_groq.csv", index=False)
    
    # 2. Evaluate with OpenRouter (model resolved from PROVIDER_CONFIG / OPENROUTER_MODEL)
    agent_or = FraudValidatorAgent(provider="openrouter")
    results_or = agent_or.evaluate_transactions(suspicious_tx)
    results_or.to_csv(f"{output_dir}/results_openrouter.csv", index=False)
    
    print(f"\nEvaluation completed. Files saved in '{output_dir}'.")