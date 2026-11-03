"""Ask the vLLM server one question over its OpenAI-compatible API; print the answer, the tokens and the speed.

    python3 cluster/vera/ask.py --port 8000 --model Qwen/Qwen2.5-32B-Instruct   (run by vera_job.sbatch)
"""
import argparse, json, time, urllib.request

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8000)
p.add_argument("--model", required=True)
p.add_argument("--question", default="In three sentences: what does an IP router do with a packet's time-to-live field?")
a = p.parse_args()

body = {"model": a.model, "messages": [{"role": "user", "content": a.question}], "max_tokens": 300, "temperature": 0}
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
reply = json.load(urllib.request.urlopen(req, timeout=600))
seconds = time.time() - t0

usage = reply["usage"]
print("QUESTION:", a.question)
print("ANSWER:  ", reply["choices"][0]["message"]["content"])
print(f"tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out; {seconds:.1f} s; "
      f"{usage['completion_tokens'] / seconds:.1f} tokens/s")
