
import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING


logger = logging.getLogger("ai_chat_api.config.mongodb")

client: Optional[AsyncIOMotorClient] = None
database: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongo() -> None:
    """Initialize MongoDB connection pool and verify connection with ping."""
    global client, database
    mongo_url = os.getenv("MONGODB_URL", "mongodb://127.0.0.1:27017")
    db_name = os.getenv("MONGODB_DB_NAME", "chat_db")

    logger.info("Connecting to MongoDB at %s (db: %s)...", mongo_url, db_name)
    try:
        client = AsyncIOMotorClient(
            mongo_url,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )
        # Verify connection
        await client.admin.command("ping")
        database = client[db_name]
        logger.info("Successfully connected to MongoDB database '%s'", db_name)

        # Setup unique index on email for users collection
        await database.users.create_index([("email", ASCENDING)], unique=True)

        # Setup indexes on token and TTL expiry for refresh_tokens collection
        await database.refresh_tokens.create_index([("token", ASCENDING)], unique=True)
        await database.refresh_tokens.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)

        # Setup indexes for conversations collection
        await database.conversations.create_index([("thread_id", ASCENDING)], unique=True)
        await database.conversations.create_index([("user_id", ASCENDING), ("updated_at", -1)])
        logger.info("MongoDB indexes verified for 'users', 'refresh_tokens', and 'conversations'")

    except Exception as exc:
        logger.error("Failed to connect to MongoDB: %s", exc)
        database = None


async def close_mongo_connection() -> None:
    global client, database
    if client is not None:
        logger.info("Closing MongoDB connection...")
        client.close()
        client = None
        database = None
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    if database is None:
        raise RuntimeError("MongoDB is not connected. Verify your local MongoDB service is running on mongodb://127.0.0.1:27017.")
    return database


async def check_mongo_health() -> bool:
    if client is None:
        return False
    try:
        await client.admin.command("ping")
        return True
    except Exception:
        return False
