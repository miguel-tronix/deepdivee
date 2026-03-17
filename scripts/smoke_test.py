#!/usr/bin/env python3
import os
import sys
import httpx
import time


def run_smoke_test():
    """
    Hits the deployed agent's health endpoint and a test query to verify
    the service is alive and the LLM/DB connections aren't completely broken.
    Expected to be run by Jenkins, GitHub Actions, or ArgoCD post-sync hooks.
    """
    # Prefer an environment variable, fallback to localhost for local testing
    base_url = os.environ.get("DEEPDIVE_API_URL", "http://localhost:8000")

    print(f"Running smoke tests against {base_url}...")

    with httpx.Client(timeout=10.0) as client:
        # 1. Health Check
        print("1. Checking /health...")
        try:
            health_res = client.get(f"{base_url}/health")
            health_res.raise_for_status()
            print("   [OK] Health check passed.")
        except Exception as e:
            print(f"   [FAIL] Health check failed: {e}")
            sys.exit(1)

        # 2. RAG Endpoint Test (Optional/Dry-run)
        # Note: In a real environment, you might want a specific known-good payload
        # that the LLM/DB can easily return quickly without consuming too many tokens.
        print("2. Checking /api/contraindications...")
        payload = {"intervention": "smoke-test-medical-intervention"}
        try:
            # We allow higher timeout for the LLM call
            rag_res = client.post(
                f"{base_url}/api/contraindications", json=payload, timeout=30.0
            )

            # Even if the DB has no context, the endpoint should return 200 ok
            # with a fallback message.
            rag_res.raise_for_status()
            data = rag_res.json()
            if "analysis" in data:
                print("   [OK] RAG endpoint returned valid schema.")
            else:
                print("   [FAIL] RAG endpoint response missing 'analysis' field.")
                sys.exit(1)

        except Exception as e:
            print(f"   [FAIL] RAG endpoint check failed: {e}")
            sys.exit(1)

    print("All smoke tests passed successfully!")


if __name__ == "__main__":
    # Give the service a moment to start up if running concurrently
    time.sleep(2)
    run_smoke_test()
