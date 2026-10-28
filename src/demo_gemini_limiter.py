
import time
import uuid
from collections import deque
from threading import Lock

import streamlit as st

from src.demo_config import DEMO_MODE


# ==========================================
# PUBLIC DEMO LIMITS
# ==========================================

WINDOW_SECONDS = 60 * 60

# Maximum Gemini requests per visitor per hour
PER_SESSION_LIMIT = 20

# Maximum Gemini requests across the
# Streamlit application process per hour
GLOBAL_LIMIT = 100


# ==========================================
# SHARED REQUEST TRACKING
# ==========================================

_lock = Lock()

_global_requests = deque()
_session_requests = {}


class DemoGeminiLimitError(RuntimeError):
    pass


def check_demo_gemini_limit():

    # The full local application is unrestricted.
    if not DEMO_MODE:
        return

    # Create an identifier for this Streamlit session.
    if "_demo_gemini_session_id" not in st.session_state:
        st.session_state["_demo_gemini_session_id"] = (
            uuid.uuid4().hex
        )

    session_id = st.session_state[
        "_demo_gemini_session_id"
    ]

    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS

    with _lock:

        # Remove expired application-wide requests.
        while (
            _global_requests
            and _global_requests[0] <= cutoff
        ):
            _global_requests.popleft()

        # Remove expired visitor requests.
        for key in list(_session_requests):

            requests = _session_requests[key]

            while requests and requests[0] <= cutoff:
                requests.popleft()

            if not requests:
                del _session_requests[key]

        visitor_requests = _session_requests.setdefault(
            session_id,
            deque()
        )

        # Enforce individual visitor limit.
        if len(visitor_requests) >= PER_SESSION_LIMIT:
            raise DemoGeminiLimitError(
                "Demo request limit reached for this "
                "session. Please try again later."
            )

        # Enforce shared application limit.
        if len(_global_requests) >= GLOBAL_LIMIT:
            raise DemoGeminiLimitError(
                "The public demo has reached its "
                "shared Gemini request limit. "
                "Please try again later."
            )

        # Reserve one request before calling Gemini.
        visitor_requests.append(now)
        _global_requests.append(now)
