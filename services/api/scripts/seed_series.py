import os
import time
import requests
from supabase import create_client, Client
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

SERIES = [
"KXSAGAWARDCASTDRAMA",
"KXWAINWRIGHTBANANAS",
"RTWAROFTHEROHIRRIM",
"KXHOWARDSTERNANNOUNCEMENT",
"KXGOVWINOMR",
"GOOGLESHARE",
"KXCONGRESSMENTION",
"KXGRAMROTY",
"KXELECTIONMOVNJGOVD",
"TOPALBUMBYKANYE",
"KXARRESTPOWELL",
"KXFORMALNOMGAETZ",
"KXBANXUS",
"KXACQANNOUNCECHROME",
"KXMLBALMOTY",
"KXDENSNOWXMAS",
"KXTRUMPSCOTUS",
"APPLEUS",
"KXNOBELPHYSICS",
"KXDCDELD",
"NASDAQ100I",
"KXRTCAUGHTSTEALING",
"KXBBNEWARTIST",
"KXFIRSTUSOPEN",
"KXOTEWALZ",
"KXASTRONAUTS",
"KXUSOMENSINGLES",
"KXMEETPTZ",
"KXWEALTH",
"KXENGAGEMENT",
"KXUSTAKEOVER",
"KXBBGROUP",
"KXGGSTANDUP",
"KXTROPSTORM",
"KXGOVCARESIGN",
"APPRANKPAID",
"KXNETFLIXRANKSHOWRUNNERUP",
"KXSAGNOMACTRLIM",
"KXUSECDEFPOLICY",
"KXNCAAFPLAYOFF",
"KXMLBPLAYOFFS",
"KXNYFWATTEND",
"KXTRUMPSTATECOUNT",
"KXDEFAULT",
"SECLABOR",
"KXALLVOTEHEGSETH",
"KXTOPALBUMBYALEXWARREN",
"KXFRANCENA",
"KXBALDONILIVELY",
"KXLIGAPORTUGAL",
"KXEARNINGSMENTIONAVGO",
"KXRYDERCUP",
"KXRTWARFARE",
"KXTHEAMAZINGRACE",
"KXGOVNYNOMR",
"KXTOP10BILLBOARDSPOTSWICKED",
"KXFEATUREKENDRICK",
"KXDEMSWEEP",
"KXOSCARACTO",
"KXHARTMENTION",
"KXSILVERAPPROVE",
"KXNEXTSPEAKER",
"KXUAW",
"KXSBADAPPEARANCES",
"KXHARVARDTAX",
"SENATECA",
"HOUSENY19",
"PRESPARTYID",
"KXAMAFMHHA",
"KXTARIFFSSTEEL",
"KXGERPERCENT",
"KXNEWTAYLORAMA",
"KXMIDSEASONINVITATIONALB",
"KXUSOPEN",
"KXCHINALLM",
"KXMLBNLEAST",
"KXCOALITION",
"KXUSDEBT",
"KXLTCMAXY",
"KXBOUNTY2025S2",
"GDPCN",
"KXJOINBARSTOOL",
"KXWTAIT",
"KXNFLSACKRECORD",
"KXISMSERVICES",
"KXTESLADELIVERYBY",
"KXFORTNITEPROAM",
"KXSECNAVY",
"KXJLEAGUEGAME",
"KXLEAVEJANGS",
"HOUSECO8",
"KXLSUCOACH",
"KXGRAMBSSFVGAOIM",
"KXGRAMBRNBP",
"SENATE",
"KXTOPSONGFATEOFOPHELIA",
"RENTNYCY",
"MANCHIN",
"KXPAYROLLCANCEL",
"KXRECBASELINE",
"KXAGNOMTXD",
"KXNEXTTEAMMLB",
"KXNEWALBUMKATYPERRY",
"KXNCAAFPREPACK4ML",
"KXNBAALLSTARS",
"KXPUBLICSWIFTLIVELY",
"KXATTYGENCO",
"KXNFLDPOY",
"KXPRESSCONF",
"KXSEC",
"KXRTAVATARFIREANDASH",
"KXNYCCORPORATETAX",
"KXPIZZASCORE9",
"KXTURKEYPARDON",
"KXARGENTINASENATE",
"KXBRENNANCHARGE",
"KXMOSTSTEAMEDSWAG",
"KXSPOTIFYALBUMJACKBOYS2",
"KXTOPPOD",
"KXRATECUTCOUNT",
"KXACKMANLEAVE",
"KXAMAFFHHA",
"KXAWARDSCMAEOTY",
"CBVOLUME",
"DINENYC",
"KXSTARSHIPLAUNCH",
"CREDEF",
"KXCLUBWCGH",
"KXFTAPRC",
"KXSAGAWARDSUPACTO",
"KXSERIEATOP4",
"KXBTCMAX100",
"KXRAINHOUM",
"KXIPOAIRTABLE",
"KXTOKENLAUNCHUNIT",
"KXALBUMSALESCARDIB",
"KXRRF",
"KXCOUNTRYSTARLINK",
"KXOH13R",
"KXNFLREDZONEBRANDADS",
"KXLOWNYC",
"KXMCSYSTEMSHOCKTWO",
"KXAMAFFLA",
"LIZARRAGA",
"KXNFLDRAFT1ST",
"KXOSCARNOMSCORE",
"KXDEFICIT",
"KXRTMEGALOPOLIS",
"KXTIME",
"KXSTARSHIP25",
"KXDRAKESUIT",
"KXPERFORMSUPERBOWL",
"KXALPHAARENAMONK",
"KXINFANTINOMENTION",
"KXWNBA",
"KXTOPALBUMSPOTIFY3",
"KXKY4R",
"RTBADBOYS",
"KXSTARWARS",
"KXUFCLHEAVYWEIGHTTITLE",
"KXUCLBTTS",
"KXXAIREASON",
"KXIPHONERELEASE",
"KXEARNINGSMENTIONCRM",
"KXITASUPERCUPADVANCE",
"KX10Y3M",
"KXMAYORJERSEY",
"KXPRESTURKEYR1ADVANCE",
"KXNFLWINS-CIN",
"KXGLACIER",
"KXSPOTIFYSONGSBTHEREFOREIAM",
"GOVPARTYWY",
"KXDEEP",
"KXMOSTWINSEMMYS",
"KXEARNINGSMENTIONBLK",
"KXATTYGENWI",
"PRESPARTYNV",
"GOVPARTYCT",
"KXCA22PRIMARY",
"KXIMPEACHCOUNT",
"KXSTABLEYIELDNOTION",
"KXBELGIANPL",
"KXAVAXMINY",
"KXTIKTOKSTORE",
"PRESPARTYWV",
"KXNEXTTEAMVERSTAPPEN",
"KXEMMYCACTO",
"KXABRAHAMSY",
"KXGRAMBDEA",
"KXBTCMAX100INAUG",
"KXSPOTIFY100MILLIONSWIFT",
"KXEARNINGSMENTIONROKU",
"KXH5N1",
"KXDKNGMUP",
"KXVOSVSTRUMP",
"HOUSEPA1",
"RTGLADIATOR2",
"KXSPOTIFYW",
"KXALTHOFFMENTION",
"KXNFLWINS-DEN",
]

SERIES_CATEGORY = {
  "KXSAGAWARDCASTDRAMA": "Entertainment",
"KXWAINWRIGHTBANANAS": "Politics",
"RTWAROFTHEROHIRRIM": "Entertainment",
"KXHOWARDSTERNANNOUNCEMENT": "Entertainment",
"KXGOVWINOMR": "Politics",
"GOOGLESHARE": "Companies",
"KXCONGRESSMENTION": "Mentions",
"KXGRAMROTY": "Entertainment",
"KXELECTIONMOVNJGOVD": "Elections",
"TOPALBUMBYKANYE": "Entertainment",
"KXARRESTPOWELL": "Politics",
"KXFORMALNOMGAETZ": "Politics",
"KXBANXUS": "Politics",
"KXACQANNOUNCECHROME": "Financials",
"KXMLBALMOTY": "Sports",
"KXDENSNOWXMAS": "Climate and Weather",
"KXTRUMPSCOTUS": "Politics",
"APPLEUS": "Companies",
"KXNOBELPHYSICS": "Entertainment",
"KXDCDELD": "Elections",
"NASDAQ100I": "Financials",
"KXRTCAUGHTSTEALING": "Entertainment",
"KXBBNEWARTIST": "Entertainment",
"KXFIRSTUSOPEN": "Sports",
"KXOTEWALZ": "Politics",
"KXASTRONAUTS": "Science and Technology",
"KXUSOMENSINGLES": "Sports",
"KXMEETPTZ": "Politics",
"KXWEALTH": "Economics",
"KXENGAGEMENT": "Entertainment",
"KXUSTAKEOVER": "Politics",
"KXBBGROUP": "Entertainment",
"KXGGSTANDUP": "Entertainment",
"KXTROPSTORM": "Climate and Weather",
"KXGOVCARESIGN": "Politics",
"APPRANKPAID": "Entertainment",
"KXNETFLIXRANKSHOWRUNNERUP": "Entertainment",
"KXSAGNOMACTRLIM": "Entertainment",
"KXUSECDEFPOLICY": "Politics",
"KXNCAAFPLAYOFF": "Sports",
"KXMLBPLAYOFFS": "Sports",
"KXNYFWATTEND": "Politics",
"KXTRUMPSTATECOUNT": "Politics",
"KXDEFAULT": "Politics",
"SECLABOR": "Politics",
"KXALLVOTEHEGSETH": "Politics",
"KXTOPALBUMBYALEXWARREN": "Entertainment",
"KXFRANCENA": "Elections",
"KXBALDONILIVELY": "Entertainment",
"KXLIGAPORTUGAL": "Sports",
"KXEARNINGSMENTIONAVGO": "Companies",
"KXRYDERCUP": "Sports",
"KXRTWARFARE": "Entertainment",
"KXTHEAMAZINGRACE": "Entertainment",
"KXGOVNYNOMR": "Politics",
"KXTOP10BILLBOARDSPOTSWICKED": "Entertainment",
"KXFEATUREKENDRICK": "Entertainment",
"KXDEMSWEEP": "Elections",
"KXOSCARACTO": "Entertainment",
"KXHARTMENTION": "Mentions",
"KXSILVERAPPROVE": "Politics",
"KXNEXTSPEAKER": "Politics",
"KXUAW": "Transportation",
"KXSBADAPPEARANCES": "Entertainment",
"KXHARVARDTAX": "Politics",
"SENATECA": "Politics",
"HOUSENY19": "Politics",
"PRESPARTYID": "Politics",
"KXAMAFMHHA": "Entertainment",
"KXTARIFFSSTEEL": "Politics",
"KXGERPERCENT": "Elections",
"KXNEWTAYLORAMA": "Entertainment",
"KXMIDSEASONINVITATIONALB": "Sports",
"KXUSOPEN": "Sports",
"KXCHINALLM": "Science and Technology",
"KXMLBNLEAST": "Sports",
"KXCOALITION": "Politics",
"KXUSDEBT": "Economics",
"KXLTCMAXY": "Crypto",
"KXBOUNTY2025S2": "Sports",
"GDPCN": "World",
"KXJOINBARSTOOL": "Companies",
"KXWTAIT": "Sports",
"KXNFLSACKRECORD": "Sports",
"KXISMSERVICES": "Economics",
"KXTESLADELIVERYBY": "Companies",
"KXFORTNITEPROAM": "Entertainment",
"KXSECNAVY": "Politics",
"KXJLEAGUEGAME": "Sports",
"KXLEAVEJANGS": "Companies",
"HOUSECO8": "Politics",
"KXLSUCOACH": "Sports",
"KXGRAMBSSFVGAOIM": "Entertainment",
"KXGRAMBRNBP": "Entertainment",
"SENATE": "Politics",
"KXTOPSONGFATEOFOPHELIA": "Entertainment",
"RENTNYCY": "Economics",
"MANCHIN": "Politics",
"KXPAYROLLCANCEL": "Economics",
"KXRECBASELINE": "Politics",
"KXAGNOMTXD": "Politics",
"KXNEXTTEAMMLB": "Sports",
"KXNEWALBUMKATYPERRY": "Entertainment",
"KXNCAAFPREPACK4ML": "Sports",
"KXNBAALLSTARS": "Sports",
"KXPUBLICSWIFTLIVELY": "Entertainment",
"KXATTYGENCO": "Elections",
"KXNFLDPOY": "Sports",
"KXPRESSCONF": "Politics",
"KXSEC": "Politics",
"KXRTAVATARFIREANDASH": "Entertainment",
"KXNYCCORPORATETAX": "Politics",
"KXPIZZASCORE9": "Sports",
"KXTURKEYPARDON": "Politics",
"KXARGENTINASENATE": "Elections",
"KXBRENNANCHARGE": "Politics",
"KXMOSTSTEAMEDSWAG": "Entertainment",
"KXSPOTIFYALBUMJACKBOYS2": "Entertainment",
"KXTOPPOD": "Entertainment",
"KXRATECUTCOUNT": "Economics",
"KXACKMANLEAVE": "Politics",
"KXAMAFFHHA": "Entertainment",
"KXAWARDSCMAEOTY": "Entertainment",
"CBVOLUME": "Companies",
"DINENYC": "World",
"KXSTARSHIPLAUNCH": "Science and Technology",
"CREDEF": "Economics",
"KXCLUBWCGH": "Sports",
"KXFTAPRC": "Politics",
"KXSAGAWARDSUPACTO": "Entertainment",
"KXSERIEATOP4": "Sports",
"KXBTCMAX100": "Crypto",
"KXRAINHOUM": "Climate and Weather",
"KXIPOAIRTABLE": "Economics",
"KXTOKENLAUNCHUNIT": "Crypto",
"KXALBUMSALESCARDIB": "Entertainment",
"KXRRF": "Politics",
"KXCOUNTRYSTARLINK": "Politics",
"KXOH13R": "Elections",
"KXNFLREDZONEBRANDADS": "Sports",
"KXLOWNYC": "Climate and Weather",
"KXMCSYSTEMSHOCKTWO": "Entertainment",
"KXAMAFFLA": "Entertainment",
"LIZARRAGA": "Politics",
"KXNFLDRAFT1ST": "Sports",
"KXOSCARNOMSCORE": "Entertainment",
"KXDEFICIT": "Politics",
"KXRTMEGALOPOLIS": "Entertainment",
"KXTIME": "Entertainment",
"KXSTARSHIP25": "Science and Technology",
"KXDRAKESUIT": "Entertainment",
"KXPERFORMSUPERBOWL": "Entertainment",
"KXALPHAARENAMONK": "Sports",
"KXINFANTINOMENTION": "Mentions",
"KXWNBA": "Sports",
"KXTOPALBUMSPOTIFY3": "Entertainment",
"KXKY4R": "Elections",
"RTBADBOYS": "Entertainment",
"KXSTARWARS": "Entertainment",
"KXUFCLHEAVYWEIGHTTITLE": "Sports",
"KXUCLBTTS": "Sports",
"KXXAIREASON": "Science and Technology",
"KXIPHONERELEASE": "Companies",
"KXEARNINGSMENTIONCRM": "Companies",
"KXITASUPERCUPADVANCE": "Sports",
"KX10Y3M": "Financials",
"KXMAYORJERSEY": "Elections",
"KXPRESTURKEYR1ADVANCE": "Elections",
"KXNFLWINS-CIN": "Sports",
"KXGLACIER": "Politics",
"KXSPOTIFYSONGSBTHEREFOREIAM": "Entertainment",
"GOVPARTYWY": "Politics",
"KXDEEP": "Entertainment",
"KXMOSTWINSEMMYS": "Entertainment",
"KXEARNINGSMENTIONBLK": "Companies",
"KXATTYGENWI": "Elections",
"PRESPARTYNV": "Politics",
"GOVPARTYCT": "Politics",
"KXCA22PRIMARY": "Elections",
"KXIMPEACHCOUNT": "Politics",
"KXSTABLEYIELDNOTION": "Crypto",
"KXBELGIANPL": "Sports",
"KXAVAXMINY": "Crypto",
"KXTIKTOKSTORE": "Politics",
"PRESPARTYWV": "Politics",
"KXNEXTTEAMVERSTAPPEN": "Sports",
"KXEMMYCACTO": "Entertainment",
"KXABRAHAMSY": "Politics",
"KXGRAMBDEA": "Entertainment",
"KXBTCMAX100INAUG": "Crypto",
"KXSPOTIFY100MILLIONSWIFT": "Entertainment",
"KXEARNINGSMENTIONROKU": "Companies",
"KXH5N1": "Health",
"KXDKNGMUP": "Companies",
"KXVOSVSTRUMP": "Politics",
"HOUSEPA1": "Politics",
"RTGLADIATOR2": "Entertainment",
"KXSPOTIFYW": "Entertainment",
"KXALTHOFFMENTION": "Mentions",
"KXNFLWINS-DEN": "Sports",
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
