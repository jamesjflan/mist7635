#%%
import pandas as pd
from geopy.geocoders import Nominatim
import folium
from geopy.extra.rate_limiter import RateLimiter
import time


# Load data
data = pd.read_excel(f'./data/full_data_set.xlsx')

# Aggregate data
player_counts = data.groupby(['Hometown', 'State']).size().reset_index(name='Count')

#%%
# Initialize geocoder
geolocator = Nominatim(user_agent="player_map")
#%%
def geocode_with_retry(address, attempt=1, max_attempts=3):
    try:
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=10)
        return geocode(address, timeout=10)  # Increase timeout to 10 seconds
    except GeocoderUnavailable as e:
        if attempt <= max_attempts:
            time.sleep(10)  # Wait 10 seconds before retrying
            return geocode_with_retry(address, attempt + 1, max_attempts)
        else:
            raise e

# Update your get_lat_lon function to use geocode_with_retry
def get_lat_lon(row):
    try:
        location = geocode_with_retry(f"{row['Hometown']}, {row['State']}, USA")
        if location:
            return pd.Series([location.latitude, location.longitude])
    except Exception as e:
        return pd.Series([None, None])

# Apply geocoding
player_counts[['Latitude', 'Longitude']] = player_counts.apply(get_lat_lon, axis=1)
#%%
# Create map
map = folium.Map(location=[37.0902, -95.7129], zoom_start=5)

# Add markers
for idx, row in player_counts.dropna().iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=row['Count'], # Adjust size based on player count
        popup=f"{row['Hometown']}, {row['State']}: {row['Count']}",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(map)

# Save map
map.save(f'./map/player_map.html')

# %%
