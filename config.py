# Alpaca Configuration
import os

# Try to get keys from Environment Variables first (Standard Alpaca Names)
API_KEY = os.getenv("APCA_API_KEY_ID", "PKHEZUOYEZEBBJWOBXCN7V4YP2")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "CTnF8JhT1EyZprLDmAoi2xjZTbFTZwsMdxGhh9Pg9hUy")

# Base URL
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets") # Use "https://api.alpaca.markets" for live trading
