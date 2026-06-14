"""Interactive terminal chat with the discovery agent — the fastest way to feel
the real conversation (the web UI isn't wired for the discovery flow yet).

Run:  venv/bin/python tests/chat.py

Commands:
  /reset     start a fresh conversation (clears memory)
  /filters   show the filters gathered so far
  /raw       toggle the agent's internal debug logs
  /quit      exit
"""
import io
import os
import sys
import uuid
import warnings
import contextlib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from agent import (get_response, user_memories, user_search_histories,
                       user_pending_queries, user_discovery_states)

C_AGENT, C_YOU, C_DIM, C_RST = "\033[96m", "\033[93m", "\033[90m", "\033[0m"


def new_session():
    return f"chat_{uuid.uuid4().hex[:8]}"


def reset(uid):
    for store in (user_memories, user_search_histories, user_pending_queries, user_discovery_states):
        store.pop(uid, None)


def main():
    uid = new_session()
    show_raw = False
    print(f"{C_DIM}Zameen discovery chat — /reset /filters /raw /quit. Session {uid}{C_RST}")
    last = {}
    while True:
        try:
            msg = input(f"{C_YOU}you ›{C_RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            reset(uid)
            uid = new_session()
            print(f"{C_DIM}— fresh session {uid} —{C_RST}")
            continue
        if msg == "/raw":
            show_raw = not show_raw
            print(f"{C_DIM}raw logs {'ON' if show_raw else 'OFF'}{C_RST}")
            continue
        if msg == "/filters":
            print(f"{C_DIM}{last.get('accumulated_filters', {})}{C_RST}")
            continue

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            r = get_response(msg, user_id=uid, channel="web")
        last = r
        if show_raw and buf.getvalue().strip():
            print(C_DIM + buf.getvalue().strip() + C_RST)

        print(f"{C_AGENT}agent ›{C_RST} {r['response']}")
        stage = r.get("stage")
        count = r.get("match_count")
        listings = r.get("listings", [])
        tag = f"[{stage}"
        if count is not None:
            tag += f" · {count} match"
        tag += "]"
        print(f"{C_DIM}{tag}{C_RST}")
        if listings:
            for l in listings[:5]:
                m = l.get("metadata", {})
                print(f"{C_DIM}   • {str(m.get('title','')).title()} — {m.get('price','')} — {m.get('location','')}{C_RST}")
        print()


if __name__ == "__main__":
    main()
