"""
FastAPI micro‑service:
• User auth & licence table – Supabase
• Usage metering  – Stripe billing
Run:  uvicorn admin_server:app --reload
"""
import os, time, stripe, supabase
from fastapi import FastAPI, HTTPException, Depends, Header

stripe.api_key      = os.getenv("STRIPE_SECRET_KEY")
SUPABASE_URL       = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE   = os.getenv("SUPABASE_SERVICE_ROLE")
sb                 = supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE)

app = FastAPI(title="CityOfficialsBot SaaS Admin")

def get_user(api_key:str = Header(...)):
    """Simple token auth"""
    row = sb.table("api_keys").select("*").eq("key",api_key).single().execute()
    if not row.data:  raise HTTPException(403,"Bad key")
    return row.data

# ----------------- endpoints -----------------------------------------------
@app.get("/quota")
def quota(user=Depends(get_user)):
    return {"remaining": user["quota_remaining"]}

@app.post("/deduct")
def deduct(amount:int, user=Depends(get_user)):
    if user["quota_remaining"] < amount:
        raise HTTPException(402,"Quota exceeded")
    sb.table("api_keys").update({"quota_remaining":user["quota_remaining"]-amount}) \
                       .eq("id", user["id"]).execute()
    return {"ok":True}

@app.post("/webhook/stripe")
def stripe_webhook(payload: dict):
    """Stripe webhook (subscribe, cancel, invoice.paid) → update Supabase"""
    # validate sig omitted
    event = payload["type"]
    sub   = payload["data"]["object"]
    if event == "invoice.paid":
        api_key = sub["metadata"]["api_key"]
        sb.table("api_keys").update({"quota_remaining": 10_000}).eq("key",api_key).execute()
    return {"received":True}
