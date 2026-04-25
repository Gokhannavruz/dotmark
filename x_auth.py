import secrets
import hashlib
import base64
import httpx
import os
from urllib.parse import urlencode
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://127.0.0.1:8000/auth/callback")


def generate_code_verifier():
    return secrets.token_urlsafe(32)


def generate_code_challenge(verifier: str):
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def get_auth_url(redirect_uri: str = None):
    state = secrets.token_urlsafe(16)
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri or REDIRECT_URI,
        "scope": "tweet.read users.read bookmark.read offline.access",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    url = f"https://twitter.com/i/oauth2/authorize?{urlencode(params)}"
    return url, state, code_verifier


async def exchange_code_for_token(code: str, code_verifier: str, redirect_uri: str = None) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.twitter.com/2/oauth2/token",
            data={
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri or REDIRECT_URI,
                "code_verifier": code_verifier,
            },
            auth=(CLIENT_ID, CLIENT_SECRET),
        )
        response.raise_for_status()
        return response.json()


async def get_user_info(access_token: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.twitter.com/2/users/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        response.raise_for_status()
        data = response.json()
        return {"id": data["data"]["id"], "username": data["data"]["username"]}
