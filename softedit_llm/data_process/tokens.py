"""Credentials for GPT-4o data generation. Read from the environment.

Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT before running the
ProtoQA generality builders.
"""
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
azure_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
