**What Phase 1 Does**

Phase 1 is the first working connection between your API and an AI model:

1. The app receives a query at `POST /v1/query/stream`.
2. It sends the query to Gemini.
3. It returns the answer as it is generated, using **Server-Sent Events (SSE)**. The client can display each token as it arrives instead of waiting for the complete answer.
4. It also sends events for a citation, time to first token, completion time, and errors.

That flow is implemented in `streaming_api.py`. The app loads environment variables and registers the Phase 1 route in `main.py`.

The other Phase 1 module, `token_economics.py`, estimates token counts, model costs, and latency. It also maps request intents to model tiers such as `EXTRACTION`, `STANDARD`, and `FRONTIER`.

**What You’ve Achieved**

You’ve put several important AI application building blocks in place:

- A FastAPI endpoint that calls a real LLM provider.
- Streaming responses, a common pattern for chat applications.
- A structured event format for tokens, metrics, citations, completion, and errors.
- Basic measurements such as time to first token and total response time.
- A starting model for comparing the cost and speed of different model tiers.

That’s meaningful progress: you’ve moved beyond just calling a model and started thinking about how an AI feature behaves as part of an application.

One important limitation: Phase 1’s citation is currently a placeholder, and the prompt doesn’t yet receive actual filing text. So it streams an AI answer, but it is **not yet answering from verified financial documents**. The cost and latency estimates are also rules of thumb, and the tier selection isn’t wired into the streaming request yet.


--------------------------------------------------------------------------------------------------------------------
** run and test phase 1 query.

The Phase 1 streaming endpoint is `POST http://127.0.0.1:8000/v1/query/stream`. It needs a valid Gemini API key because it makes a real call to Gemini.

1. Add your key to the project’s `.env` file:
   GEMINI_API_KEY=your_key_here

2. Start the app from the project root if it isn’t already running:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
   ```

3. In a second PowerShell window, send a sample request. Use `curl.exe` so PowerShell doesn’t substitute its `curl` alias, and `-N` to show the stream as it arrives:

   ```powershell
   curl.exe -N -X POST "http://127.0.0.1:8000/v1/query/stream" `
     -H "Content-Type: application/json" `
     -d '{"query":"What should an investor look for when reviewing a company'\''s operating cash flow?","document_id":"sample-10k"}'
   ```

You should see events like `citation`, `metric`, multiple `token` events, and finally `done`. The `document_id` is optional; the current citation is a placeholder and the query is not yet grounded in an actual SEC filing. If the request reports a model provider error, check that the key is valid and available to the running app.