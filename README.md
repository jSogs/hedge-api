# Hedge API

A FastAPI-based service that provides intelligent hedge recommendations for prediction markets (Kalshi) using AI-powered semantic search and personalized risk analysis.

## Overview

Hedge API helps individuals and businesses identify relevant prediction market hedges based on their risk profiles. The system uses vector embeddings and LLMs to match user profiles with Kalshi events, monitor world news for significant events, and generate personalized hedging recommendations.

## Features

- 🔍 **Semantic Search**: Vector-based search for Kalshi events using OpenAI embeddings
- 🎯 **Personalized Recommendations**: AI-powered hedge recommendations based on user risk profiles
- 📰 **News Monitoring**: Automated monitoring of world events and automatic recommendation generation
- 🔔 **Notifications**: Real-time notifications for new recommendations triggered by news events
- 💰 **Price Tracking**: Automated price updates from Kalshi API
- 🧠 **LLM-Powered Analysis**: GPT-4 powered analysis for recommendation explanations

## Tech Stack

- **Framework**: FastAPI
- **Database**: Supabase (PostgreSQL with pgvector)
- **AI/ML**: OpenAI (GPT-4o-mini, text-embedding-3-small)
- **News API**: NewsAPI.org
- **Market Data**: Kalshi API

## Setup

### Prerequisites

- Python 3.8+
- Supabase account and project
- OpenAI API key
- NewsAPI key (optional, for news monitoring)

### Installation

1. **Clone the repository**
   ```bash
   cd /path/to/hedge-api
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd services/api
   pip install fastapi uvicorn supabase openai python-dotenv requests
   ```

4. **Set up environment variables**

   Create a `.env` file in `services/api/`:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
   OPENAI_API_KEY=your_openai_api_key
   NEWS_API_KEY=your_newsapi_key  # Optional, for news monitoring
   ```

5. **Database Setup**

   Run the SQL migrations in your Supabase project to create the necessary tables:
   - `kalshi_events` (with embedding vector column)
   - `markets`
   - `market_outcomes`
   - `market_prices`
   - `profiles`
   - `recommendations`
   - `news_events`
   - `news_event_recommendations`
   - `notifications`

   See the SQL schema section below for table definitions.

## Running the API

### Development Server

```bash
cd services/api
python3 -m uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Scripts

All scripts are located in `services/api/scripts/` and can be run with:

```bash
cd services/api
python3 scripts/<script_name>.py
```

### Available Scripts

#### `seed_series.py`
Seeds Kalshi events and markets from specified series tickers.

#### `backfill_event_embeddings.py`
Batch processes events missing embeddings. Processes in batches of 64.

#### `update_prices.py`
Fetches current prices from Kalshi API and updates the database.

#### `monitor_news.py`
Monitors news for significant world events and automatically creates recommendations for affected users.

**Setup scheduler** (runs every 15 minutes):
```bash
crontab -e

# Add this line:
*/15 * * * * cd /path/to/hedge-api/services/api && python3 scripts/monitor_news.py >> /tmp/monitor_news.log 2>&1
```

#### `get_series.py`
Utility script to extract categories from Kalshi series data.

#### `process_new_events.py`
Processes newly added events: embeds them and checks for user recommendations.

**Usage:**
```bash
# Process events from last hour
python3 scripts/process_new_events.py --hours 1

# Process specific event IDs
python3 scripts/process_new_events.py --event-ids uuid1 uuid2 uuid3
```

**What it does:**
1. Embeds any events missing embeddings
2. Searches for users whose profiles match the new events
3. Creates recommendations and notifications for matching users

**Use this after manually inserting events/markets** to automatically:
- Generate embeddings for new events
- Check if any users should be notified about new hedging opportunities

## API Endpoints

### Search

**POST** `/v1/search`
- Search for Kalshi events using semantic search
- Request body:
  ```json
  {
    "query": "inflation rates",
    "limit": 5,
    "markets_per_event": 3,
    "min_similarity": 0.5
  }
  ```

### Recommendations

**POST** `/v1/recommendations/run`
- Generate personalized hedge recommendations for a user
- Request body:
  ```json
  {
    "user_id": "uuid",
    "limit": 10,
    "match_count": 20,
    "markets_per_event": 3
  }
  ```

### Notifications

**GET** `/v1/notifications/{user_id}`
- Get notifications for a user
- Query params: `?unread_only=true&limit=50`

**POST** `/v1/notifications/{notification_id}/read`
- Mark a notification as read

**POST** `/v1/notifications/{user_id}/read-all`
- Mark all notifications as read for a user

**GET** `/v1/notifications/{user_id}/unread-count`
- Get count of unread notifications

### Chat

**POST** `/v1/chat/message`
- Send a message and get AI response with optional market data
- Request body:
  ```json
  {
    "conversation_id": "uuid (optional, creates new if not provided)",
    "message": "What should I hedge against?",
    "user_id": "uuid"
  }
  ```
- Response includes text response and optional market data

**GET** `/v1/chat/conversations/{user_id}`
- List user's conversations with message counts

**GET** `/v1/chat/{conversation_id}`
- Get full conversation history with all messages

**DELETE** `/v1/chat/{conversation_id}?user_id=uuid`
- Delete a conversation and all its messages

**PATCH** `/v1/chat/{conversation_id}/title?user_id=uuid&title=New Title`
- Update conversation title

## Database Schema

### Key Tables

- **`kalshi_events`**: Prediction market events with vector embeddings
- **`markets`**: Individual markets within events
- **`market_outcomes`**: YES/NO outcomes for binary markets
- **`market_prices`**: Time-series price data
- **`profiles`**: User risk profiles (region, industry, sensitivities, etc.)
- **`recommendations`**: Generated hedge recommendations
- **`news_events`**: Tracked news articles with embeddings
- **`news_event_recommendations`**: Links news events to recommendations
- **`notifications`**: User notifications

### Required Supabase Functions

- **`search_kalshi_events_with_markets`**: RPC function for vector similarity search
  - Parameters: `query_embedding` (vector(1536)), `match_count` (int), `markets_per_event` (int)
  - Returns: JSON with matching events, markets, outcomes, and prices

## Architecture

### Recommendation Flow

1. User profile is loaded from database
2. Profile attributes are converted to a query string
3. Query is embedded using OpenAI
4. Vector similarity search finds matching Kalshi events
5. LLM (GPT-4o-mini) selects best hedges and explains why
6. Recommendations are persisted to database

### News Monitoring Flow

1. Script fetches recent news from NewsAPI
2. LLM classifies importance (0-1 score, threshold: 0.7)
3. Important news is embedded
4. Vector search finds affected Kalshi events
5. For each user profile, recommendations are generated
6. Notifications are created for users

### Vector Search

Uses pgvector extension in PostgreSQL for efficient similarity search. Embeddings are 1536-dimensional vectors from OpenAI's `text-embedding-3-small` model.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | Yes |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `NEWS_API_KEY` | NewsAPI key for news monitoring | Optional |

## Development

### Project Structure

```
hedge-api/
├── services/
│   └── api/
│       ├── app/
│       │   ├── ai/          # AI/embedding utilities
│       │   ├── db/          # Database client
│       │   ├── routers/     # API route handlers
│       │   └── main.py      # FastAPI app
│       └── scripts/         # Batch/admin scripts
└── README.md
```

### Adding New Features

1. Create router in `app/routers/`
2. Register router in `app/main.py`
3. Add any required database migrations
4. Update this README

## Rate Limiting

The system includes rate limiting workarounds for OpenAI API:
- Exponential backoff retry logic
- Delays between API calls
- Batch processing where possible
- Pre-filtering to reduce unnecessary LLM calls

## Troubleshooting

### "No news items found"
- Check NewsAPI key is set correctly
- Verify date format in `fetch_recent_news()` matches NewsAPI requirements
- Try using `/top-headlines` endpoint instead of `/everything`

### Rate limit errors
- Increase delays between API calls
- Reduce number of articles processed per run
- Check OpenAI usage limits

### Import errors
- Ensure virtual environment is activated
- Install all dependencies: `pip install -r requirements.txt` (if available)
- Check Python path in scripts

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

