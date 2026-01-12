import os
import time
import requests
from supabase import create_client, Client
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

SERIES = [
    "KXGRETAGAZA",
"KXATTYGENNC",
"KXINJH",
"KXWHATSONSTAGEBPIAM",
"KXELECTIONMOVNJGOV",
"TOPALBUMRECORDY",
"KXCHOPINCOMPETITION",
"KXENDORSEMAM",
"KXSCREWWORMCOUNT",
"KXTRUMPGOLF",
"KXMAYOROAKLAND",
"KXSNAPRESTRICT",
"KXSTUDENTVISAS",
"KXMOLDOVAMAJORITY",
"KXOKSTCOACH",
"KXNBADRAFT1",
"KXBRAZILTARIFFSIZE",
"KXNFLEXACTWINSCAR",
"BAFTASUPACTO",
"KXMOVIESITP",
"CREDEFMINMAX",
"KXTENCOACH",
"KXSENATECHILE",
"INXI",
"KXNYCCOUCNIL8",
"KXLOSEREALMADRID",
"KXBILLBOARDPEAKHOUDINI",
"KXCOSTARICAPRES",
"KXCZECHCOALITION",
"KXRANKLISTAIWFCIS",
"KXEOTRUMPTERM",
"KXITAXH",
"KXTOP10SABRINAC",
"KXCODINGMODEL",
"KXDOGE",
"RENTNYCM",
"KXMITACKMAN",
"KXEASTCOASTPORTSTRIKE",
"KXLANAREY",
"KXCAPCONTROL",
"KXSTARWARS",
"KXNC01R",
"KXMEXICOCITY",
"KXDEBTGROWTH",
"KXSPOTIFYCHARTSTREAKMANCHILD",
"KXMAYORMIA",
"KXHIGHNY0",
"KXEARNINGSMENTIONDIS",
"KXBIGBROTHER",
"KXSTEFANIKCOUNT",
"KXTACAPORTADVANCE",
"KXAPPRANKFREESECOND",
"KXNFLWINS-CLE",
]

SERIES_CATEGORY = {
  "KXGRETAGAZA": "Politics",
"KXATTYGENNC": "Elections",
"KXINJH": "Politics",
"KXWHATSONSTAGEBPIAM": "Entertainment",
"KXELECTIONMOVNJGOV": "Elections",
"TOPALBUMRECORDY": "Entertainment",
"KXCHOPINCOMPETITION": "Entertainment",
"KXENDORSEMAM": "Elections",
"KXSCREWWORMCOUNT": "Health",
"KXTRUMPGOLF": "Sports",
"KXMAYOROAKLAND": "Elections",
"KXSNAPRESTRICT": "Politics",
"KXSTUDENTVISAS": "Politics",
"KXMOLDOVAMAJORITY": "Elections",
"KXOKSTCOACH": "Sports",
"KXNBADRAFT1": "Sports",
"KXBRAZILTARIFFSIZE": "Politics",
"KXNFLEXACTWINSCAR": "Sports",
"BAFTASUPACTO": "Entertainment",
"KXMOVIESITP": "Entertainment",
"CREDEFMINMAX": "Economics",
"KXTENCOACH": "Sports",
"KXSENATECHILE": "Elections",
"INXI": "Financials",
"KXNYCCOUCNIL8": "Politics",
"KXLOSEREALMADRID": "Sports",
"KXBILLBOARDPEAKHOUDINI": "Entertainment",
"KXCOSTARICAPRES": "Elections",
"KXCZECHCOALITION": "Politics",
"KXRANKLISTAIWFCIS": "Entertainment",
"KXEOTRUMPTERM": "Politics",
"KXITAXH": "Politics",
"KXTOP10SABRINAC": "Entertainment",
"KXCODINGMODEL": "Science and Technology",
"KXDOGE": "Crypto",
"RENTNYCM": "Economics",
"KXMITACKMAN": "Politics",
"KXEASTCOASTPORTSTRIKE": "Economics",
"KXLANAREY": "Entertainment",
"KXCAPCONTROL": "Elections",
"KXSTARWARS": "Entertainment",
"KXNC01R": "Elections",
"KXMEXICOCITY": "Politics",
"KXDEBTGROWTH": "Politics",
"KXSPOTIFYCHARTSTREAKMANCHILD": "Entertainment",
"KXMAYORMIA": "Elections",
}

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_events_for_series(series_ticker: str, max_retries: int = 5):
    events = []
    cursor = None

    while True:
        params = {
            "limit": 200,
            "series_ticker": series_ticker.upper(),
            "with_nested_markets": "true",
            "status": "open",
        }
        if cursor:
            params["cursor"] = cursor

        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                r = requests.get(f"{BASE_URL}/events", params=params, timeout=30)
                
                # Handle rate limiting
                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", 5))
                    wait_time = min(retry_after, 2 ** attempt)  # Exponential backoff, max from header
                    print(f"  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                
                r.raise_for_status()
                payload = r.json()
                break
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    print(f"  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Failed to fetch {series_ticker} after {attempt + 1} attempts")
                    raise
            except Exception as e:
                print(f"  Error fetching {series_ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        batch = payload.get("events", [])
        events.extend(batch)

        cursor = payload.get("cursor")
        if not cursor:
            break

        # Add delay between paginated requests
        time.sleep(0.5)

    return events

def _parse_strike_date(e: dict) -> str | None:
    """
    Kalshi event payloads can vary. Try common keys, then fall back to market close/resolve date.
    """
    # Common possibilities
    for k in ["strike_date", "strikeDate", "settlement_date", "resolve_date"]:
        v = e.get(k)
        if v:
            try:
                # If it's already YYYY-MM-DD
                return datetime.fromisoformat(v[:10]).date().isoformat()
            except Exception:
                pass

    # Fall back: use earliest market close_time / resolve_time date if present
    markets = e.get("markets") or []
    times = []
    for m in markets:
        for tk in ["close_time", "resolve_time"]:
            tv = m.get(tk)
            if tv:
                times.append(tv)
    if times:
        try:
            # ISO string; just take date part
            return datetime.fromisoformat(sorted(times)[0].replace("Z", "+00:00")).date().isoformat()
        except Exception:
            return None

    return None


def upsert_events(events: list[dict], series_ticker: str) -> dict[str, str]:
    """
    Upserts kalshi_events rows, returns mapping:
      external_event_id -> internal_uuid (kalshi_events.id)
    """
    if not events:
        return {}

    category = SERIES_CATEGORY.get(series_ticker)

    rows = []
    for e in events:
        ext_event_id = e.get("ticker") or e.get("event_ticker") or e.get("id")
        if not ext_event_id:
            continue

        title = e.get("title") or e.get("name") or ""
        subtitle = e.get("subtitle") or e.get("sub_title")

        rows.append({
            "platform": "kalshi",
            "external_event_id": ext_event_id,
            "series_ticker": e.get("series_ticker") or series_ticker,
            "title": title,
            "subtitle": subtitle,
            "description": e.get("description"),
            "category": category,
            "region": "US",
            "status": e.get("status"),
            "created_time": e.get("created_time"),
            "updated_time": e.get("updated_time"),
            "strike_date": _parse_strike_date(e),
            "event_json": e,
        })

    sb.table("kalshi_events").upsert(
        rows,
        on_conflict="platform,external_event_id"
    ).execute()

    ext_ids = [r["external_event_id"] for r in rows]
    res = sb.table("kalshi_events") \
        .select("id,external_event_id") \
        .eq("platform", "kalshi") \
        .in_("external_event_id", ext_ids) \
        .execute()

    return {row["external_event_id"]: row["id"] for row in (res.data or [])}



def upsert_markets(events: list[dict], event_id_map: dict[str, str]) -> dict[str, str]:
    """
    Upserts markets rows, returns mapping:
      external_market_id -> internal_uuid (markets.id)
    """
    market_rows = []

    for e in events:
        ext_event_id = e.get("ticker") or e.get("event_ticker") or e.get("id")
        internal_event_id = event_id_map.get(ext_event_id)
        if not internal_event_id:
            continue

        for m in (e.get("markets") or []):
            ext_market_id = m.get("ticker") or m.get("market_ticker") or m.get("id")
            if not ext_market_id:
                continue

            market_rows.append({
                "event_id": internal_event_id,
                "platform": "kalshi",
                "external_market_id": ext_market_id,
                "title": m.get("title") or "",
                "description": m.get("description"),
                "market_type": m.get("type") or m.get("market_type"),
                "category": None,
                "region": "US",
                "close_time": m.get("close_time"),
                "resolve_time": m.get("resolve_time"),
                "url": m.get("url"),
                "is_active": True,
                "market_json": m,
                # "tags": [],
                # "keywords": [],
            })

    if not market_rows:
        return {}

    # Upsert on unique(platform, external_market_id)
    sb.table("markets").upsert(
        market_rows,
        on_conflict="platform,external_market_id"
    ).execute()

    # Re-fetch IDs for outcomes linkage
    ext_market_ids = list({r["external_market_id"] for r in market_rows})
    res = sb.table("markets") \
        .select("id,external_market_id") \
        .eq("platform", "kalshi") \
        .in_("external_market_id", ext_market_ids) \
        .execute()

    mapping = {row["external_market_id"]: row["id"] for row in (res.data or [])}
    return mapping


def upsert_outcomes(events: list[dict], market_id_map: dict[str, str]) -> int:
    """
    Upserts YES/NO outcomes (or multiple if provided).
    Returns number of outcome rows written (attempted).
    """
    outcome_rows = []

    for e in events:
        for m in (e.get("markets") or []):
            ext_market_id = m.get("ticker") or m.get("market_ticker") or m.get("id")
            internal_market_id = market_id_map.get(ext_market_id)
            if not internal_market_id:
                continue

            # Kalshi sometimes provides outcomes, otherwise assume YES/NO
            outs = m.get("outcomes")

            if isinstance(outs, list) and len(outs) > 0:
                for o in outs:
                    label = o.get("title") or o.get("label") or o.get("name")
                    if not label:
                        continue
                    outcome_rows.append({
                        "market_id": internal_market_id,
                        "label": label,
                        "external_outcome_id": o.get("id") or o.get("outcome_id"),
                        "outcome_json": o,
                    })
            else:
                # Default to YES/NO for binary markets
                outcome_rows.append({
                    "market_id": internal_market_id,
                    "label": "YES",
                    "external_outcome_id": "YES",
                    "outcome_json": {},
                })
                outcome_rows.append({
                    "market_id": internal_market_id,
                    "label": "NO",
                    "external_outcome_id": "NO",
                    "outcome_json": {},
                })

    if not outcome_rows:
        return 0

    # Upsert on unique(market_id, label)
    sb.table("market_outcomes").upsert(
        outcome_rows,
        on_conflict="market_id,label"
    ).execute()

    return len(outcome_rows)


def main():
    total_events = 0
    total_markets = 0
    total_outcomes = 0
    failed_series = []

    print(f"Processing {len(SERIES)} series with rate limiting...")
    
    for i, s in enumerate(SERIES):
        print(f"\n[{i+1}/{len(SERIES)}] Processing {s}...")
        
        try:
            evts = fetch_events_for_series(s)
            print(f"  Fetched {len(evts)} events")
            
            if not evts:
                print(f"  No events found for {s}")
                continue
            
            total_events += len(evts)

            # 1) Events
            event_id_map = upsert_events(evts, s)

            # 2) Markets
            market_id_map = upsert_markets(evts, event_id_map)

            # 3) Outcomes
            n_out = upsert_outcomes(evts, market_id_map)

            n_markets = len(market_id_map)
            total_markets += n_markets
            total_outcomes += n_out

            print(f"  ✓ Upserted events={len(event_id_map)}, markets={n_markets}, outcomes={n_out}")
            
            # Add delay between series to avoid rate limiting
            if i < len(SERIES) - 1:  # Don't sleep after last one
                time.sleep(1)
                
        except Exception as e:
            print(f"  ✗ Failed to process {s}: {e}")
            failed_series.append(s)
            # Continue with next series instead of crashing
            time.sleep(2)  # Wait longer after error
            continue

    print(f"\n{'='*60}")
    print(f"DONE totals:")
    print(f"  Events: {total_events}")
    print(f"  Markets: {total_markets}")
    print(f"  Outcomes: {total_outcomes}")
    
    if failed_series:
        print(f"\n⚠ Failed series ({len(failed_series)}):")
        for s in failed_series:
            print(f"  - {s}")
        print("\nYou can retry these later.")


if __name__ == "__main__":
    main()
